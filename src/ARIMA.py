import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from math import sqrt
import os

# Cargar y preparar datos
file_path = r'C:\Users\LAPT\Documents\JOHN\UDLA\2. CIENCIA DE DATOS Y BUSSINES INTELIGENCE\8. PROYECTO DE GRADO\BASE DE DATOS PROYECTO UDLA\data\ventas_limpias_completas.xlsx'
df = pd.read_excel(file_path)
df['FECHA'] = pd.to_datetime(df['FECHA'])
df = df.groupby('FECHA')['CANTIDAD'].sum().asfreq('D').fillna(0)

# Agrupar por semana y sumar cantidades
serie_semanal = df.resample('W').sum()

# Dividir en entrenamiento y prueba (últimas 10 semanas para prueba)
train = serie_semanal[:-10]
test = serie_semanal[-10:]

# Ajustar modelo ARIMA
modelo_arima = ARIMA(train, order=(2, 1, 1))
resultado = modelo_arima.fit()

# Predicciones
predicciones = resultado.forecast(steps=len(test))

# Evaluación
mae = mean_absolute_error(test, predicciones)
rmse = sqrt(mean_squared_error(test, predicciones))
mape = mean_absolute_percentage_error(test, predicciones)
r2 = r2_score(test, predicciones)

# Crear gráfico
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Entrenamiento', color='blue')
plt.plot(test.index, test, label='Real', color='orange')
plt.plot(test.index, predicciones, label='Predicción ARIMA(2,1,1)', color='green', linestyle='dashed')
plt.title("Modelo ARIMA(2,1,1) – Predicción semanal")
plt.xlabel("Fecha")
plt.ylabel("Cantidad vendida")
plt.legend()
plt.tight_layout()
plt.grid(True)

# Guardar imagen
os.makedirs('images', exist_ok=True)
plt.savefig('images/prediccion_ARIMA_semanal.png')
plt.show()

# Resumen
summary = {
    'Modelo': ['ARIMA(2,1,1)'],
    'MAE': [mae],
    'RMSE': [rmse],
    'MAPE': [mape],
    'R²': [r2]
}
summary_df = pd.DataFrame(summary)
print(summary_df)




