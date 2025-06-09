import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from math import sqrt
import os
import warnings
warnings.filterwarnings("ignore")

# Crear carpeta para imagen
os.makedirs("images", exist_ok=True)

# Cargar y preparar datos
df = pd.read_excel(r'C:\Users\LAPT\Documents\JOHN\UDLA\2. CIENCIA DE DATOS Y BUSSINES INTELIGENCE\8. PROYECTO DE GRADO\BASE DE DATOS PROYECTO UDLA\data\ventas_limpias_completas.xlsx')
df['FECHA'] = pd.to_datetime(df['FECHA'])

# Agrupar por semana
df_semanal = df.resample('W', on='FECHA')['CANTIDAD'].sum()

# Dividir en entrenamiento y prueba
train = df_semanal[:-10]
test = df_semanal[-10:]

# Ajustar modelo AR
model_ar = AutoReg(train, lags=7, old_names=False)
model_fit = model_ar.fit()

# Predicción
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

# Evaluación
mape = mean_absolute_percentage_error(test, predictions)
rmse = sqrt(mean_squared_error(test, predictions))

# Comparación en DataFrame
comparison_df = pd.DataFrame({
    'Fecha': test.index,
    'Real': test.values,
    'Pronosticado': predictions.values
}).set_index('Fecha')

print(comparison_df)
print(f"MAPE: {mape:.4f}")
print(f"RMSE: {rmse:.4f}")

# Gráfico
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Entrenamiento', color='blue')
plt.plot(test.index, test, label='Real', color='orange', linestyle='--', marker='o')
plt.plot(test.index, predictions, label='Pronosticado AR', color='green', linestyle='--', marker='x')
plt.xlabel('Fecha')
plt.ylabel('Cantidad Vendida')
plt.title('Pronóstico con modelo AR (AutoReg)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('images/comparacion_AR_forecast.png')
plt.show()
