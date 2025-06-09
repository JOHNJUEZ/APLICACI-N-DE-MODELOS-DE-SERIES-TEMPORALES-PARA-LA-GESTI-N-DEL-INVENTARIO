import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from math import sqrt
import os

# 1. Cargar datos
file_path = r'C:\Users\LAPT\Documents\JOHN\UDLA\2. CIENCIA DE DATOS Y BUSSINES INTELIGENCE\8. PROYECTO DE GRADO\BASE DE DATOS PROYECTO UDLA\data\ventas_limpias_completas.xlsx'
df = pd.read_excel(file_path)

# 2. Asegurar formato de fecha
df['FECHA'] = pd.to_datetime(df['FECHA'])

# 3. Agrupar por semana (sin forzar frecuencia diaria antes)
df_semanal = df.resample('W', on='FECHA')['CANTIDAD'].sum()

# 4. División en entrenamiento y prueba
train = df_semanal[:-10]
test = df_semanal[-10:]

# 5. Ajustar modelo ARIMA
modelo_arima = ARIMA(train, order=(2, 1, 1))
resultado = modelo_arima.fit()

# 6. Predicción
predicciones = resultado.forecast(steps=len(test))

# 7. Evaluación
mae = mean_absolute_error(test, predicciones)
rmse = sqrt(mean_squared_error(test, predicciones))
mape = mean_absolute_percentage_error(test, predicciones)
r2 = r2_score(test, predicciones)

# 8. Mostrar métricas
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.5f}, R²: {r2:.2f}")

# 9. Gráfico
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Entrenamiento', color='blue')
plt.plot(test.index, test, label='Real', color='orange')
plt.plot(test.index, predicciones, label='Predicción ARIMA(2,1,1)', color='green', linestyle='dashed')
plt.title("Modelo ARIMA(2,1,1) – Predicción semanal")
plt.xlabel("Fecha")
plt.ylabel("Cantidad vendida")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 10. Guardar gráfico
os.makedirs('images', exist_ok=True)
plt.savefig('images/prediccion_ARIMA_semanal.png')
plt.show()

# 11. Resumen en DataFrame
summary = {
    'Modelo': ['ARIMA(2,1,1)'],
    'MAE': [mae],
    'RMSE': [rmse],
    'MAPE': [mape],
    'R²': [r2]
}
summary_df = pd.DataFrame(summary)
print(summary_df)
