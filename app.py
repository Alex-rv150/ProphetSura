import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import plotly.express as px
import holidays
import itertools
import matplotlib.pyplot as plt

st.set_page_config(page_title="Predicción con Prophet", layout="wide")

# 📁 Cargar dataset
@st.cache_data
def cargar_datos():
    df = pd.read_csv("Prophet 2021-2024 (F).csv")  # <--- CAMBIA AQUÍ con el nombre real del CSV
    df["ds"] = pd.to_datetime(df["ds"])
    return df

df = cargar_datos()

# 🧹 Dividir datos
df_entrenamiento = df[df["ds"] < "2024-04-30"]
df_prueba = df[df["ds"] >= "2024-04-30"]

# 📌 Extraer municipios y servicios
municipios = sorted(set(i.split(" - ")[0] for i in df["unique_id"]))
servicios = sorted(set(i.split(" - ")[1] for i in df["unique_id"]))

# 🎛️ Interfaz
st.title("🔮 Predicción por municipio y servicio")
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
    st.stop()

# 📅 Festivos
holidays_df = pd.DataFrame(
    holidays.CO(years=[2025, 2024, 2023, 2022, 2021]).items(),
    columns=["ds", "holiday"]
)

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
