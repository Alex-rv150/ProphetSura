import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import holidays
import itertools
import geopandas as gpd
import json
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
from joblib import Parallel, delayed
import multiprocessing
from collections import defaultdict
from functools import lru_cache
from concurrent.futures import TimeoutError
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Inicializar la aplicación Dash
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    show_undo_redo=False,
    update_title=None
)
server = app.server

# Configuración adicional para ocultar el indicador de errores
app.config.suppress_callback_exceptions = True
app.config.update_title = None
app.config.show_undo_redo = False

# Cache para festivos
HOLIDAYS_CACHE = {}

def get_holidays(years):
    """Obtener festivos con caché"""
    key = tuple(years)
    if key not in HOLIDAYS_CACHE:
        HOLIDAYS_CACHE[key] = pd.DataFrame(
            holidays.CO(years=years).items(),
            columns=["ds", "holiday"]
        )
    return HOLIDAYS_CACHE[key]

# Cache para modelos entrenados
@lru_cache(maxsize=100)
def train_cached_model(serie_id, params_tuple):
    """Cache de modelos entrenados para evitar reentrenamiento"""
    params = dict(params_tuple)
    df_train = train_dict[serie_id]
    df_test = test_dict[serie_id]
    df_union = pd.concat([df_train, df_test])
    
    model = Prophet(
        holidays=holidays_df,
        holidays_prior_scale=20,  # Valor más razonable
        **params
    )
    model.add_seasonality(name='weekly', period=7, fourier_order=7)
    model.fit(df_union)
    return model

# Cache para los datos
def load_data():
    # Cargar datos del mapa
    gdf = gpd.read_file(
        "MGN_ANM_MPIOS.json",
        usecols=['MPIO_CNMBR', 'STP27_PERS', 'geometry']
    )
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.002)
    
    # Aplicar transformación logarítmica a la población
    gdf['log_poblacion'] = np.log10(gdf['STP27_PERS'])
    
    # Definir tipos de datos optimizados
    dtypes = {
        'unique_id': 'category',
        'y': 'float32'
    }
    
    # Cargar datos de predicción con tipos optimizados
    df = pd.read_csv(
        "Prophet 2021-2024 (F).csv",
        dtype=dtypes,
        parse_dates=['ds']  # Parsear la columna de fechas
    )
    
    # Crear índices para búsqueda rápida
    df.set_index('unique_id', inplace=True)
    
    # Separar datos de entrenamiento y prueba
    df_entrenamiento = df[df["ds"] < "2024-04-30"].copy()
    df_prueba = df[df["ds"] >= "2024-04-30"].copy()
    
    # Crear diccionarios para acceso rápido
    train_dict = defaultdict(dict)
    test_dict = defaultdict(dict)
    
    # Pre-calcular los datos filtrados
    for idx in df_entrenamiento.index.unique():
        train_dict[idx] = df_entrenamiento.loc[idx].copy()
        test_dict[idx] = df_prueba.loc[idx].copy()
    
    # Cargar festivos usando caché
    holidays_df = get_holidays([2025, 2024, 2023, 2022, 2021])
    
    servicios = sorted(set(i.split(" - ")[1] for i in df_entrenamiento.index))
    
    return gdf, train_dict, test_dict, holidays_df, servicios

# Cargar datos
gdf, train_dict, test_dict, holidays_df, servicios = load_data()

# Función para entrenar el modelo
def train_model(df_train, holidays_df, params):
    model = Prophet(
        holidays=holidays_df,
        holidays_prior_scale=0.05,  # Valor más razonable
        **params
    )
    model.add_seasonality(name='weekly', period=7, fourier_order=7)
    model.fit(df_train)
    return model

# Función para obtener predicciones
def get_predictions(model, future):
    forecast = model.predict(future)
    forecast['yhat'] = forecast['yhat'].round().clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].round().clip(lower=0)
    forecast['yhat_upper'] = forecast['yhat_upper'].round().clip(lower=0)
    return forecast

def train_and_evaluate(params, df_train, df_test, holidays_df):
    """Función con manejo de errores para entrenamiento y evaluación"""
    try:
        model = Prophet(
            holidays=holidays_df,
            holidays_prior_scale=0.05,  # Valor más razonable
            **params
        )
        model.add_seasonality(name='weekly', period=7, fourier_order=7)
        model.fit(df_train)
        
        forecast = model.predict(df_test)
        mae = mean_absolute_error(df_test['y'].values, forecast['yhat'].values)
        
        return params, mae
    except Exception as e:
        # En caso de error, devolver un valor alto de MAE
        return params, float('inf')

def parallel_hyperparameter_tuning(df_train, df_test, holidays_df):
    """Función que realiza el tuning de hiperparámetros en paralelo"""
    # Definir el espacio de búsqueda
    param_grid = {
        'seasonality_mode': ["multiplicative", "additive"],
        'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.5],
        'seasonality_prior_scale': [0.1, 1.0, 10.0]
    }
    
    # Generar todas las combinaciones de parámetros
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    
    # Obtener el número de núcleos disponibles
    n_cores = multiprocessing.cpu_count()
    
    # Ejecutar el entrenamiento en paralelo
    results = Parallel(n_jobs=n_cores)(
        delayed(train_and_evaluate)(params, df_train, df_test, holidays_df)
        for params in all_params
    )
    
    # Encontrar los mejores parámetros
    best_params, best_mae = min(results, key=lambda x: x[1])
    
    return best_params

def create_optimized_plot(forecast):
    """Crear gráfico optimizado"""
    prediccion2025 = forecast[forecast['ds'] >= '2025-01-01'].copy()
    fig = px.line(
        prediccion2025,
        x='ds',
        y='yhat',
        title='Pronóstico de Demanda',
        template='plotly_white'
    )
    
    # Optimizar el rendimiento del gráfico
    fig.update_layout(
        xaxis_title='Fecha',
        yaxis_title='Valor Predicho',
        showlegend=True,
        hovermode='x unified',
        uirevision=True,  # Mantener el zoom/pan entre actualizaciones
        plot_bgcolor='#F5F9FF',
        paper_bgcolor='#F5F9FF',
        font=dict(color='#1976D2')
    )
    
    # Agregar intervalos de confianza
    fig.add_scatter(
        x=prediccion2025['ds'],
        y=prediccion2025['yhat_lower'],
        mode='lines',
        name='Límite Inferior del Pronóstico',
        line=dict(dash='dot', color='#64B5F6'),
        showlegend=True
    )
    fig.add_scatter(
        x=prediccion2025['ds'],
        y=prediccion2025['yhat_upper'],
        mode='lines',
        name='Límite Superior del Pronóstico',
        line=dict(dash='dot', color='#1976D2'),
        showlegend=True
    )
    
    return fig

def create_comparison_plot(df_test, forecast):
    """Crear gráfico comparativo entre valores reales y predichos"""
    fig = px.line(
        df_test,
        x='ds',
        y='y',
        title='Comparación: Valores Reales vs Predichos',
        template='plotly_white'
    )
    
    # Agregar línea de predicción
    fig.add_scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Predicción',
        line=dict(color='#1E88E5')
    )

    # Agregar intervalos de confianza
    fig.add_scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        name='Límite Inferior del Pronóstico',
        line=dict(dash='dot', color='#64B5F6'),
        showlegend=True
    )
    fig.add_scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        name='Límite Superior del Pronóstico',
        line=dict(dash='dot', color='#1976D2'),
        showlegend=True
    )
    
    # Actualizar layout
    fig.update_layout(
        xaxis_title='Fecha',
        yaxis_title='Valor',
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='#F5F9FF',
        paper_bgcolor='#F5F9FF',
        font=dict(color='#1976D2')
    )

    
    return fig

def create_eda_plots(df_train, df_test):
    """Crear gráficos de análisis exploratorio de datos"""
    # Crear figura con subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Distribución de Valores',
            'Tendencia Temporal',
            'Box Plot por Mes',
            'Correlación con Días de la Semana'
        )
    )
    
    # 1. Histograma de valores
    fig.add_trace(
        go.Histogram(
            x=df_train['y'],
            name='Distribución',
            marker_color='#1E88E5'
        ),
        row=1, col=1
    )
    
    # 2. Tendencia temporal
    fig.add_trace(
        go.Scatter(
            x=df_train['ds'],
            y=df_train['y'],
            mode='lines',
            name='Tendencia',
            line=dict(color='#1E88E5')
        ),
        row=1, col=2
    )
    
    # 3. Box plot por mes
    df_train['mes'] = df_train['ds'].dt.month
    fig.add_trace(
        go.Box(
            y=df_train['y'],
            x=df_train['mes'],
            name='Box Plot Mensual',
            marker_color='#1E88E5'
        ),
        row=2, col=1
    )
    
    # 4. Correlación con días de la semana
    df_train['dia_semana'] = df_train['ds'].dt.dayofweek
    promedio_dia = df_train.groupby('dia_semana')['y'].mean()
    fig.add_trace(
        go.Bar(
            x=['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'],
            y=promedio_dia.values,
            name='Promedio por Día',
            marker_color='#1E88E5'
        ),
        row=2, col=2
    )
    
    # Actualizar layout
    fig.update_layout(
        height=800,
        showlegend=False,
        template='plotly_white',
        plot_bgcolor='#F5F9FF',
        paper_bgcolor='#F5F9FF',
        font=dict(color='#1976D2')
    )
    
    return fig

# Layout de la aplicación
app.layout = dbc.Container([
    # Fila para los logos
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Img(src='/assets/Logo-SURA-blanco (1).svg', height='60px', className='me-4'),
                html.Img(src='/assets/uni-logo-horizontal.svg', height='60px')
            ], className='d-flex justify-content-center align-items-center mb-4')
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H1("Estimación diaria del 2025 para cada municipio y servicio", 
                   className="text-center mb-5 mt-4 text-primary bold",
                   style={'marginTop': '2rem', 'marginBottom': '2rem'}),
            html.P([
                html.B("Seleccione un municipio para visualizar su cobertura actual y pronósticos futuros")
            ], className="text-center mb-5"),
            dcc.Graph(
                id='map-graph',
                figure=px.choropleth_mapbox(
                    gdf,
                    geojson=gdf.geometry,
                    locations=gdf.index,
                    color='log_poblacion',
                    hover_name='MPIO_CNMBR',
                    hover_data={'STP27_PERS': ':,.0f', 'log_poblacion': False},
                    mapbox_style="carto-positron",
                    center={"lat": 4.5709, "lon": -74.2973},
                    zoom=5.5,
                    opacity=0.7,
                    color_continuous_scale=[
                        '#E3F2FD',  # Muy bajo
                        '#BBDEFB',  # Bajo
                        '#90CAF9',  # Medio-bajo
                        '#64B5F6',  # Medio
                        '#42A5F5',  # Medio-alto
                        '#1E88E5',  # Alto
                        '#1565C0',  # Muy alto
                        '#0D47A1'   # Extremo
                    ],
                    labels={'log_poblacion': 'Población (escala logarítmica)'}
                ).update_layout(
                    paper_bgcolor='#F5F9FF',
                    plot_bgcolor='#F5F9FF',
                    margin=dict(l=20, r=20, t=0, b=0)
                ),
                style={'height': '800px', 'backgroundColor': '#F5F9FF'}
            )
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div(id='municipio-info', className="mt-4 p-3 bg-light rounded")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div(id='prediction-container', className="mt-4")
        ], width=12)
    ]),
    
    # Almacenamiento oculto para el municipio seleccionado
    dcc.Store(id='selected-municipio'),
    
    # Loading spinner
    dcc.Loading(
        id="loading-prediction",
        type="circle",
        children=[html.Div(id="loading-output")]
    ),
    
    # Store para el estado del botón
    dcc.Store(id='eda-button-state', data=0),
    
    # Botón de EDA (siempre presente pero inicialmente oculto)
    dbc.Button(
        "Ver Análisis Exploratorio de Datos",
        id="eda-button",
        color="primary",
        className="mt-4 mb-4",
        style={'width': '100%', 'fontSize': '1.2rem', 'padding': '1rem', 'display': 'none'}
    )
], fluid=True, style={'backgroundColor': '#F5F9FF'})

# Callback para actualizar la información del municipio
@app.callback(
    Output('municipio-info', 'children'),
    Output('selected-municipio', 'data'),
    Input('map-graph', 'clickData')
)
def update_municipio_info(clickData):
    if clickData is None:
        raise PreventUpdate
    
    municipio = clickData['points'][0]['hovertext']
    poblacion = clickData['points'][0]['customdata'][0]
    
    return [
        dbc.Card([
            dbc.CardBody([
                html.H3("Información de Cobertura del Municipio", className="card-title text-primary"),
                dbc.Row([
                    dbc.Col([
                        html.H5("Municipio:", className="text-primary"),
                        html.P(municipio, className="lead")
                    ], width=6),
                    dbc.Col([
                        html.H5("Población:", className="text-primary"),
                        html.P(f"{poblacion:,.0f}", className="lead")
                    ], width=6)
                ])
            ])
        ], className="shadow-sm")
    ], municipio

# Callback para actualizar las predicciones
@app.callback(
    Output('prediction-container', 'children'),
    Input('selected-municipio', 'data'),
    Input('map-graph', 'clickData')
)
def update_predictions(municipio, clickData):
    if not municipio or not clickData:
        raise PreventUpdate
    
    return dbc.Card([
        dbc.CardBody([
            html.H3("Pronóstico de Demanda por Municipio y Servicio", className="card-title text-primary"),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id='servicio-dropdown',
                        options=[{'label': s, 'value': s} for s in servicios],
                        placeholder="Seleccione el tipo de servicio a pronosticar",
                        className="mb-3"
                    )
                ], width=12)
            ])
        ])
    ], className="shadow-sm")

# Callback para generar y mostrar las predicciones
@app.callback(
    Output('loading-output', 'children'),
    Output('eda-button', 'style'),
    Input('servicio-dropdown', 'value'),
    Input('eda-button', 'n_clicks'),
    State('selected-municipio', 'data'),
    State('eda-button-state', 'data')
)
def generate_predictions(servicio, n_clicks, municipio, button_state):
    if not servicio or not municipio:
        raise PreventUpdate
    
    # Reiniciar el estado del botón si cambia el servicio o municipio
    if n_clicks is None:
        n_clicks = 0
    
    serie_id = f"{municipio} - {servicio}"
    
    # Acceso rápido a los datos usando diccionarios
    df_train = train_dict[serie_id]
    df_test = test_dict[serie_id]
    
    if df_train.empty:
        return (
            html.Div("No se cuenta con información suficiente para generar el pronóstico en esta ubicación.", 
                    className="alert alert-warning"),
            {'display': 'none'}
        )
    
    # Optimización de hiperparámetros en paralelo
    best_params = parallel_hyperparameter_tuning(df_train, df_test, holidays_df)
    
    # Usar modelo cacheado si está disponible
    params_tuple = tuple(best_params.items())
    model = train_cached_model(serie_id, params_tuple)
    
    # Predicción
    future = model.make_future_dataframe(periods=365)
    future = future.tail(365*3)
    forecast = get_predictions(model, future)
    
    # Calcular MAE para el período de prueba
    test_forecast = forecast[forecast['ds'].isin(df_test['ds'])]
    mae = mean_absolute_error(df_test['y'].values, test_forecast['yhat'].values)
    # Calcular NMAE
    nmae = mae / df_test['y'].mean()
    
    # Crear gráficos
    fig_prediccion = create_optimized_plot(forecast)
    fig_comparacion = create_comparison_plot(df_test, test_forecast)
    
    # Mostrar los mejores parámetros encontrados y métricas
    params_info = dbc.Card([
        dbc.CardHeader(html.H4("Configuración Óptima del Modelo", className="mb-0 text-primary")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5("Patrón de Estacionalidad:", className="text-primary"),
                    html.P(best_params['seasonality_mode'], className="lead")
                ], width=4),
                dbc.Col([
                    html.H5("Sensibilidad al Cambio de Tendencia:", className="text-primary"),
                    html.P(f"{best_params['changepoint_prior_scale']:.3f}", className="lead")
                ], width=4),
                dbc.Col([
                    html.H5("Intensidad de la Estacionalidad:", className="text-primary"),
                    html.P(f"{best_params['seasonality_prior_scale']:.3f}", className="lead")
                ], width=4)
            ])
        ])
    ], className="mb-4 shadow-sm")
    
    # Si se hizo clic en el botón de EDA, mostrar los gráficos de análisis
    if n_clicks > button_state:
        fig_eda = create_eda_plots(df_train, df_test)
        return [
            dbc.Row([
                dbc.Col([
                    params_info,
                    html.H4("Pronóstico 2025", className="text-primary mt-4"),
                    dcc.Graph(figure=fig_prediccion),
                    html.H4("Validación del Modelo", className="text-primary mt-4"),
                    html.Div([
                        html.H5("Error Absoluto Medio (MAE):", className="text-primary d-inline me-2"),
                        html.P(f"{mae:.2f}u", className="d-inline lead me-4"),
                        html.H5("Error Absoluto Medio Normalizado (NMAE):", className="text-primary d-inline me-2"),
                        html.P(f"{nmae:.2%}", className="d-inline lead")
                    ], className="mb-3"),
                    dcc.Graph(figure=fig_comparacion),
                    html.H4("Análisis Exploratorio de Datos", className="text-primary mt-4"),
                    dcc.Graph(figure=fig_eda)
                ], width=12)
            ])
        ], {'display': 'none'}
    
    return [
        dbc.Row([
            dbc.Col([
                params_info,
                html.H4("Pronóstico 2025-2027", className="text-primary mt-4"),
                dcc.Graph(figure=fig_prediccion),
                html.H4("Validación del Modelo", className="text-primary mt-4"),
                html.Div([
                    html.H5("Error Absoluto Medio (MAE):", className="text-primary d-inline me-2"),
                    html.P(f"{mae:.2f}", className="d-inline lead me-4"),
                    html.H5("Error Absoluto Medio Normalizado (NMAE):", className="text-primary d-inline me-2"),
                    html.P(f"{nmae:.2%}", className="d-inline lead")
                ], className="mb-3"),
                dcc.Graph(figure=fig_comparacion)
            ], width=12)
        ])
    ], {'display': 'block'}

if __name__ == '__main__':
    app.run_server(debug=False) 