# estacionalidad.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# Asegurarse de que exista la carpeta 'images'
os.makedirs('images', exist_ok=True)

# Cargar los datos
df = pd.read_excel(r'C:\Users\LAPT\Documents\JOHN\UDLA\2. CIENCIA DE DATOS Y BUSSINES INTELIGENCE\8. PROYECTO DE GRADO\BASE DE DATOS PROYECTO UDLA\data\ventas_limpias_completas.xlsx')
df['FECHA'] = pd.to_datetime(df['FECHA'])
serie = df.groupby('FECHA')['CANTIDAD'].sum().asfreq('D').fillna(0)

# Prueba Dickey-Fuller Aumentada
resultado_adf = adfuller(serie)
print("Resultado ADF:")
print(f"Estadístico: {resultado_adf[0]:.4f}")
print(f"p-valor: {resultado_adf[1]:.4e}")
for clave, valor in resultado_adf[4].items():
    print(f"Valor crítico ({clave}): {valor:.3f}")
print("¿Serie estacionaria?", "Sí" if resultado_adf[1] < 0.05 else "No")

# Descomposición estacional
descomposicion = seasonal_decompose(serie, model='additive')
fig = descomposicion.plot()
fig.set_size_inches(10, 8)
plt.tight_layout()
plt.savefig('images/descomposicion_estacional.png')
plt.close()

