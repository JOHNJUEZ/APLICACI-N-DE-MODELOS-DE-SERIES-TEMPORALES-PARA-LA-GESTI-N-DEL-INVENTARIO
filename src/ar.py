import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from math import sqrt
import os
import warnings
warnings.filterwarnings("ignore")

# Asegurar que la carpeta para guardar imágenes exista
os.makedirs("images", exist_ok=True)

# Cargar datos limpios
df = pd.read_excel(r'C:\Users\LAPT\Documents\JOHN\UDLA\2. CIENCIA DE DATOS Y BUSSINES INTELIGENCE\8. PROYECTO DE GRADO\BASE DE DATOS PROYECTO UDLA\data\ventas_limpias_completas.xlsx')
df['FECHA'] = pd.to_datetime(df['FECHA'])
df = df.groupby('FECHA')['CANTIDAD'].sum().reset_index()

# Filtrar datos desde julio 2024 hasta mayo 2025
df_filtered = df[(df['FECHA'] >= '2024-07-01') & (df['FECHA'] <= '2025-05-31')]
df_filtered.set_index('FECHA', inplace=True)
serie = df_filtered['CANTIDAD'].asfreq('D').fillna(0)

# Dividir en entrenamiento y prueba
train = serie[:-30]
test = serie[-30:]

# Ajustar modelo AR
model_ar = AutoReg(train, lags=7, old_names=False)
model_fit = model_ar.fit()

# Predicción
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

# Calcular MAPE y RMSE
mape = mean_absolute_percentage_error(test, predictions)
rmse = sqrt(mean_squared_error(test, predictions))

# Crear DataFrame para comparar
comparison_df = pd.DataFrame({
    'Fecha': test.index,
    'Real': test.values,
    'Pronosticado': predictions.values
}).set_index('Fecha')

print(comparison_df)
print(f"MAPE: {mape:.4f}")
print(f"RMSE: {rmse:.4f}")

# Graficar comparación
plt.figure(figsize=(12, 6))
plt.plot(comparison_df.index, comparison_df['Real'], label='Real', color='blue', linestyle='--', marker='o')
plt.plot(comparison_df.index, comparison_df['Pronosticado'], label='Pronosticado', color='orange', linestyle='-', marker='x')
plt.xlabel('Fecha')
plt.ylabel('Cantidad')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('images/comparacion_AR_forecast.png')
plt.show()
