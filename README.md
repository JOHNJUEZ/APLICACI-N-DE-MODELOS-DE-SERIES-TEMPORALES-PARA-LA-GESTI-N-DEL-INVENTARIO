# 📊 Predicción de la Demanda en Microempresas – Caso Lomo Fino

Este repositorio contiene el desarrollo completo del proyecto de tesis titulado:

**“Aplicación de modelos de series temporales para la gestión del inventario en microempresas: Caso Lomo Fino”**

## 🎯 Objetivo del proyecto

Aplicar modelos estadísticos (ARIMA, SARIMA, SARIMAX) para anticipar la demanda mensual de productos en la microempresa ecuatoriana *Lomo Fino*, y demostrar cómo la analítica de datos puede apoyar la toma de decisiones comerciales basadas en evidencia.

---

## 📁 Estructura del repositorio

- `data/`: contiene la base de datos histórica de ventas limpias utilizadas para el análisis (2024–2025).
- `images/`: gráficos generados durante el análisis (visualizaciones, comparaciones de modelos).
- `src/`: scripts en Python utilizados para:
  - Limpieza de datos
  - Visualización exploratoria
  - Construcción y validación de modelos ARIMA, SARIMA y SARIMAX
  - Comparación de métricas

---

## ⚙️ Requisitos

- Python 3.10+
- Bibliotecas necesarias:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - statsmodels
  - pmdarima
  - scikit-learn

Puedes instalar todo con:

```bash
pip install -r requirements.txt
```

---

## 🚀 ¿Cómo ejecutar?

1. Descarga o clona el repositorio:
```bash
git clone https://github.com/JOHNJUEZ/APLICACI-N-DE-MODELOS-DE-SERIES-TEMPORALES-PARA-LA-GESTI-N-DEL-INVENTARIO.git
```

2. Ve a la carpeta `src/` y ejecuta los scripts en el siguiente orden sugerido:
   - `limpieza_datos.py`
   - `visualizacion_datos.py`
   - `seleccion_modelo.py`
   - `SARIMA.py`
   - `evaluacion_modelo.py`

3. Los resultados aparecerán en la carpeta `images/` y en consola.

---

## 📈 Resultados principales

El modelo **SARIMA** fue seleccionado como el mejor para predecir la demanda, con:
- MAE: 2.56
- RMSE: 3.24
- MAPE: 12%
- R²: 0.64

---

## 👨‍💻 Autor

**John Juez**  
Maestría en Negocios Internacionales y Ciencia de Datos – UDLA, Ecuador
