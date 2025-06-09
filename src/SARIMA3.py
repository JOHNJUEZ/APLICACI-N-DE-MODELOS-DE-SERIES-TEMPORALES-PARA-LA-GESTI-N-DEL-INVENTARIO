import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from math import sqrt

# Cargar datos
file_path = r'C:\Users\LAPT\Documents\JOHN\UDLA\2. CIENCIA DE DATOS Y BUSSINES INTELIGENCE\8. PROYECTO DE GRADO\BASE DE DATOS PROYECTO UDLA\data\ventas_limpias_completas.xlsx'
df = pd.read_excel(file_path)

# Asegurar que FECHA es datetime
df['FECHA'] = pd.to_datetime(df['FECHA'])

# Agrupar por día y luego por semana, sumando las cantidades
df_diaria = df.groupby('FECHA')['CANTIDAD'].sum()
df_semanal = df_diaria.resample('W').sum()

# Verificar los primeros valores
print(df_semanal.head(10))

# Dividir en conjunto de entrenamiento y prueba
train = df_semanal[:-10]
test = df_semanal[-10:]

# Ajustar modelo SARIMA
modelo = SARIMAX(train, order=(0,0,0), seasonal_order=(1, 0, 0, 7))
resultado = modelo.fit(disp=False)

# Predicción
predicciones = resultado.forecast(steps=10)

# Evaluación
mae = mean_absolute_error(test, predicciones)
rmse = sqrt(mean_squared_error(test, predicciones))
mape = mean_absolute_percentage_error(test, predicciones)
r2 = r2_score(test, predicciones)

# Imprimir métricas
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.5f}, R²: {r2:.2f}")

# Graficar resultados
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Entrenamiento', color='blue')
plt.plot(test.index, test, label='Real', color='orange')
plt.plot(test.index, predicciones, label='Pronóstico SARIMA', color='green', linestyle='--')
plt.title('Predicción con modelo SARIMA')
plt.xlabel('Fecha')
plt.ylabel('Cantidad Vendida')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('prediccion_SARIMA.png')
plt.show()

# Agregar a la tabla de resultados
print(pd.DataFrame({
    'Modelo': ['SARIMA(0,0,0)(1,0,0,7)'],
    'MAE': [mae],
    'RMSE': [rmse],
    'MAPE': [mape],
    'R²': [r2],
    'AIC': [resultado.aic],
    'BIC': [resultado.bic]
}))


