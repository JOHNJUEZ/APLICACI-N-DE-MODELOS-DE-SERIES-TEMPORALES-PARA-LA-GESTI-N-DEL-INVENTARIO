# prediccion_modelo.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# Asegurarse de que exista la carpeta 'images'
os.makedirs('images', exist_ok=True)

# Cargar los datos
ruta_archivo = r'C:\Users\LAPT\Documents\JOHN\UDLA\2. CIENCIA DE DATOS Y BUSSINES INTELIGENCE\8. PROYECTO DE GRADO\BASE DE DATOS PROYECTO UDLA\data\ventas_limpias_completas.xlsx'
df = pd.read_excel(ruta_archivo)
df['FECHA'] = pd.to_datetime(df['FECHA'])

# Agrupar y preparar la serie temporal de la variable endógena
serie = df.groupby('FECHA')['CANTIDAD'].sum()
serie = serie[serie > 0]  # Filtrar solo días con ventas

# Variable exógena: precio de venta diario promedio
exog = df.groupby('FECHA')['PVD'].mean()

# Alinear ambas series
datos_combinados = pd.concat([serie, exog], axis=1).dropna()
serie_alineada = datos_combinados.iloc[:, 0]
exog_alineada = datos_combinados.iloc[:, 1]

# Ajustar modelo SARIMAX
modelo = sm.tsa.statespace.SARIMAX(
    serie_alineada,
    exog=exog_alineada,
    order=(0, 0, 0),
    seasonal_order=(0, 0, 0, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)
resultado = modelo.fit(disp=False)

# Hacer predicción para los próximos 30 días usando la última observación de exógena
ultimo_valor_exog = exog_alineada.iloc[-1]
exog_pred = pd.Series([ultimo_valor_exog] * 30)
predicciones = resultado.get_forecast(steps=30, exog=exog_pred)

# Visualizar predicciones
pred_mean = predicciones.predicted_mean
pred_ci = predicciones.conf_int()

plt.figure(figsize=(12, 6))
plt.plot(serie_alineada, label='Histórico')
plt.plot(pred_mean.index, pred_mean, label='Predicción', color='green')
plt.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='lightgreen', alpha=0.5)
plt.title('Predicción de demanda diaria para 30 días')
plt.xlabel('Fecha')
plt.ylabel('Cantidad vendida')
plt.legend()
plt.tight_layout()
plt.savefig('images/prediccion_30_dias.png')
plt.show()



