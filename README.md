# 📊 Predicción de la Demanda de Servicios de Salud Laboral - ARL SURA

## 🩺 Descripción del Reto

La **previsión de la demanda de servicios de salud** es un componente clave para lograr una planificación eficiente y una asignación óptima de recursos en los sistemas sanitarios. Este desafío cobra una relevancia aún mayor en el ámbito de la **salud ocupacional**, donde los servicios relacionados con **accidentes y enfermedades laborales** presentan una naturaleza urgente, impredecible y crítica.

Este proyecto tiene como objetivo desarrollar un modelo de predicción que permita **anticipar la demanda de servicios médicos en los diferentes municipios de Colombia**, con un enfoque específico en las atenciones derivadas de riesgos laborales. La solución busca dotar a **ARL SURA** de una herramienta capaz de:

- Garantizar la **disponibilidad de recursos** en el momento y lugar indicados.
- **Optimizar la respuesta operativa** ante eventos de salud laboral.
- **Reducir el impacto negativo** de accidentes laborales mediante una atención oportuna.
- Mejorar la **eficiencia en la gestión de costos** sin sacrificar la calidad del servicio.

### 📊 Características del modelo

La solución considera múltiples factores determinantes en la demanda, tales como:

- **Patrones de estacionalidad**
- **Tendencias históricas de atención médica**
- **Características demográficas** específicas por municipio
- **Indicadores económicos locales**

### 🎯 Entregables

El entregable consiste en un modelo implementado en Python, acompañado por los archivos necesarios para su ejecución. La herramienta proporciona **predicciones detalladas por municipio y tipo de servicio médico**, con especial énfasis en los casos de **accidentes y enfermedades laborales**.

---

## ⚙️ Cómo usar el modelo

### Requisitos previos

Tener instalado IDE de su preferencia. Antes de ejecutar el archivo ´app.py´, asegurese de tener instalado Python (>= 3.8), estar en la ubicación correcta y de haber creado un entorno virtual:

```bash
python -m venv prophet_env
source prophet_env/bin/activate        # En Windows: env\Scripts\activate

```
### 🛠️ Instalación y ejecución del modelo

Instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```
Finalmente, ejecutar la aplicación:
```bash
python app_dash.py
```

### Archivos incluidos

- `app.py`: archivo principal que contiene el flujo completo de análisis, entrenamiento, validación, predicción e interfaz de la solución.
- `Prophet 2021-2024 (F).csv`: Dataset con el historial de demanda de servicios por municipio y tipo de atención (Comprimido).
- `requirements.txt`: Lista de librerías necesarias.

### Ejecución del modelo

1. Abrir el archivo `app.py` en su IDE.
2. Asegúrate de que el archivo `Prophet 2021-2024 (F).csv` esté en la misma carpeta que `app.py` y descomprimido.
3. El `app.py` incluye:
   - Limpieza y preparación de datos
   - Ingeniería de variables relevantes (estacionalidad, tendencias, etc.)
   - Entrenamiento del modelo Prophet
   - Generación de predicciones por municipio y tipo de servicio
   - Visualización de los resultados

---

## 📈 Interpretación de los resultados

El modelo basado en **Facebook Prophet** genera predicciones de la demanda futura de servicios de salud, específicamente:

- **Predicción temporal**: para cada combinación de municipio y tipo de servicio (por ejemplo, “atención por accidente laboral en Medellín”), se generan valores estimados de demanda diaria o mensual, según la granularidad del dataset.
- **Intervalos de confianza**: se incluyen bandas de predicción que permiten entender la incertidumbre asociada a la proyección.
- **Componentes del modelo**:
  - Tendencia: evolución general de la demanda a lo largo del tiempo.
  - Estacionalidad: patrones recurrentes semanales, mensuales o anuales.
  - Festivos y efectos especiales: pueden incluirse manualmente (opcional).

### Visualizaciones incluidas

- Gráficas interactivas de predicción con sus intervalos de confianza.
- Comparación del modelo con los datos conocidos.
- Análisis de componentes del modelo para comprender qué factores impulsan la demanda.


### Utilidad para la planificación

Estas predicciones pueden ser utilizadas por SURA para:

- Dimensionar el personal médico necesario por municipio.
- Prever el stock de recursos sanitarios (ambulancias, insumos, etc.).
- Detectar municipios con alta volatilidad o riesgo de saturación.
- Evaluar la necesidad de estrategias preventivas según patrones históricos.
