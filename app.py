import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import plotly.express as px
import holidays
import itertools
import matplotlib.pyplot as plt
import geopandas as gpd
import folium
import random
import branca.colormap as cm
from streamlit_folium import st_folium
import json

# Configuración de la página
st.set_page_config(
    page_title="Predicción con Prophet",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración para reducir recargas y ajustar tamaños
st.markdown("""
    <style>
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        .stButton button {width: 100%;}
        .municipio-info {
            padding: 20px;
            border-radius: 10px;
            background-color: #f0f2f6;
            margin-top: 20px;
        }
        .map-container {
            width: 100%;
            height: 400px;
            margin: 0 auto;
            position: relative;
        }
        .map-container iframe {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }
        @media (max-width: 1200px) {
            .map-container {
                height: 350px;
            }
        }
        @media (max-width: 768px) {
            .map-container {
                height: 300px;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Cache para los colores de población
@st.cache_data
def get_population_colors():
    return {
        'default': '#808080',
        'ranges': [
            (10000, '#B3E5FC'),
            (50000, '#4FC3F7'),
            (100000, '#29B6F6'),
            (200000, '#03A9F4'),
            (500000, '#0288D1'),
            (1000000, '#0277BD'),
            (2000000, '#01579B'),
            (float('inf'), '#014377')
        ]
    }

def get_color(population):
    if pd.isna(population): 
        return get_population_colors()['default']
    
    for threshold, color in get_population_colors()['ranges']:
        if population < threshold:
            return color
    return get_population_colors()['ranges'][-1][1]

# Cache para los datos del mapa
@st.cache_data(ttl=3600)
def load_municipios():
    gdf = gpd.read_file(
        "MGN_ANM_MPIOS.json",
        usecols=['MPIO_CNMBR', 'STP27_PERS', 'geometry']
    )
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.002)
    return gdf

# Cache para el mapa base
@st.cache_data
def create_base_map():
    return folium.Map(
        location=[4.5709, -74.2973],
        zoom_start=5.5,
        tiles='CartoDB positron',
        control_scale=True,
        prefer_canvas=True
    )

# Cache para los datos de predicción
@st.cache_data
def load_prediction_data():
    df = pd.read_csv("Prophet 2021-2024 (F).csv")
    df["ds"] = pd.to_datetime(df["ds"])
    return df

# Cache para los festivos
@st.cache_data
def load_holidays():
    return pd.DataFrame(
        holidays.CO(years=[2025, 2024, 2023, 2022, 2021]).items(),
        columns=["ds", "holiday"]
    )

# Precargar todos los datos
with st.spinner('Cargando datos...'):
    # Cargar datos del mapa
    gdf = load_municipios()
    
    # Cargar datos de predicción
    df = load_prediction_data()
    df_entrenamiento = df[df["ds"] < "2024-04-30"]
    df_prueba = df[df["ds"] >= "2024-04-30"]
    
    # Cargar festivos
    holidays_df = load_holidays()
    
    # Extraer municipios y servicios
    municipios = sorted(set(i.split(" - ")[0] for i in df["unique_id"]))
    servicios = sorted(set(i.split(" - ")[1] for i in df["unique_id"]))

# Inicializar estados de la sesión
if 'show_map' not in st.session_state:
    st.session_state.show_map = True  # Mostrar mapa por defecto
if 'selected_municipio' not in st.session_state:
    st.session_state.selected_municipio = None

# Mostrar el mapa
st.title("Mapa de Municipios de Colombia")
st.markdown("""
Este mapa muestra los municipios de Colombia con información de población. 
Los colores representan diferentes rangos de población.
""")

# Crear el mapa base desde cache
m = create_base_map()

# Añadir capa de municipios con estilo simplificado
geojson = folium.GeoJson(
    gdf,
    name="Municipios",
    tooltip=folium.GeoJsonTooltip(
        fields=["MPIO_CNMBR", "STP27_PERS"],
        aliases=["Municipio:", "Población:"],
        style="""
            background-color: white;
            border: 2px solid black;
            border-radius: 3px;
            box-shadow: 3px;
        """
    ),
    style_function=lambda feature: {
        'fillColor': get_color(feature['properties']['STP27_PERS']),
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.7,
    },
    highlight_function=lambda x: {
        'weight': 3,
        'fillOpacity': 0.9
    }
).add_to(m)

# Añadir control de capas
folium.LayerControl().add_to(m)

# Crear contenedor para el mapa con tamaño fijo
map_container = st.container()
with map_container:
    map_data = st_folium(
        m,
        width="100%",
        height=400,
        key="map",
        returned_objects=["last_active_drawing"]
    )

# Crear contenedor para la información del municipio
info_container = st.container()

# Actualizar información del municipio seleccionado sin recargar
if map_data and map_data.get('last_active_drawing'):
    municipio = map_data['last_active_drawing']['properties']['MPIO_CNMBR']
    poblacion = map_data['last_active_drawing']['properties']['STP27_PERS']
    
    with info_container:
        st.write("---")
        st.write("### Información del Municipio Seleccionado")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Municipio:** {municipio}")
        with col2:
            st.write(f"**Población:** {poblacion:,.0f}")

# 🎛️ Interfaz de predicción
st.write("---")
st.write("### 🔮 Predicción por municipio y servicio")
col1, col2 = st.columns(2)
with col1:
    municipio = st.selectbox("Selecciona un municipio", municipios)
with col2:
    servicio = st.selectbox("Selecciona un servicio", servicios)

serie_id = f"{municipio} - {servicio}"
st.markdown(f"**Serie seleccionada:** `{serie_id}`")

# 📊 Filtrar datos
df_train = df_entrenamiento[df_entrenamiento["unique_id"] == serie_id].copy()
df_test = df_prueba[df_prueba["unique_id"] == serie_id].copy()
df_union = pd.concat([df_train, df_test])

if df_train.empty:
    st.warning("⚠️ No hay datos suficientes para esta combinación.")
else:
    # 🧪 Tuning de hiperparámetros
    st.info("🔧 Ajustando modelo...")
    param_grid = {
        'seasonality_mode': ["multiplicative", "additive"],
        'changepoint_prior_scale': [0.01, 0.05,0.1, 0.5],
        'seasonality_prior_scale': [0.1, 1.0,  10.0]
    }
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

    maes = []
    for params in all_params:
        model = Prophet(holidays=holidays_df, holidays_prior_scale=20, **params)
        model.add_country_holidays(country_name='CO')
        model.add_seasonality(name='weekly', period=7, fourier_order=5)
        model.fit(df_train)
        forecast = model.predict(df_test)
        mae = mean_absolute_error(df_test['y'].values, forecast['yhat'].values)
        maes.append(mae)

    best_params = all_params[np.argmin(maes)]

    # 📈 Modelo final
    model = Prophet(holidays=holidays_df, holidays_prior_scale=20, **best_params)
    model.add_country_holidays(country_name='CO')
    model.add_seasonality(name='weekly', period=7, fourier_order=7)
    model.fit(df_union)

    # 🔮 Predicción
    future = model.make_future_dataframe(periods=365)
    future = future.tail(365*3)
    forecast = model.predict(future)

    # 🧼 Postprocesar
    forecast['yhat'] = forecast['yhat'].round()
    forecast['yhat'] = forecast['yhat'].clip(lower=0)

    forecast['yhat_lower'] = forecast['yhat_lower'].round()
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)

    forecast['yhat_upper'] = forecast['yhat_upper'].round()
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)

    # 📊 Gráfico interactivo
    fig = px.line(forecast, x='ds', y='yhat', title='📈 Pronóstico')
    fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Inferior', line=dict(dash='dot'))
    fig.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Superior', line=dict(dash='dot'))
    fig.update_layout(xaxis_title='Fecha', yaxis_title='Valor Predicho', template='plotly_white')

    st.plotly_chart(fig, use_container_width=True)

    # 📊 Comparación del modelo con datos de entrenamiento (in-sample)
    st.subheader("📘 Comparación en entrenamiento")

    forecast_train = model.predict(df_train)

    forecast_train['yhat'] = forecast_train['yhat'].round()
    forecast_train['yhat'] = forecast_train['yhat'].clip(lower=0)

    forecast_train['yhat_lower'] = forecast_train['yhat_lower'].round()
    forecast_train['yhat_lower'] = forecast_train['yhat_lower'].clip(lower=0)

    forecast_train['yhat_upper'] = forecast_train['yhat_upper'].round()
    forecast_train['yhat_upper'] = forecast_train['yhat_upper'].clip(lower=0)

    mae_train = mean_absolute_error(df_train["y"], forecast_train["yhat"])
    st.markdown(f"**MAE (entrenamiento)**: `{mae_train:.2f}`")

    fig_train = px.line()
    fig_train.add_scatter(x=df_train['ds'], y=df_train['y'], name="Real (entrenamiento)")
    fig_train.add_scatter(x=forecast_train['ds'], y=forecast_train['yhat'], name="Predicción (entrenamiento)")
    fig_train.update_layout(
        title="📊 Ajuste del modelo a los datos de entrenamiento",
        xaxis_title="Fecha", yaxis_title="Valor", template="plotly_white"
    )
    st.plotly_chart(fig_train, use_container_width=True)

    # 🧩 Componentes
    st.subheader("📉 Componentes del modelo")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    # 📁 Exportar
    st.download_button(
        label="📥 Descargar predicción (CSV)",
        data=forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False),
        file_name=f'prediccion_{municipio}_{servicio}.csv',
        mime='text/csv'
    )
