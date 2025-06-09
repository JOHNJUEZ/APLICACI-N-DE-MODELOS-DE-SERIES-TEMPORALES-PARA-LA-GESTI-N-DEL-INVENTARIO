# src/seleccion_parametros_sarimax.py

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

# Crear carpeta de imágenes
os.makedirs('images', exist_ok=True)

# 1. Cargar los datos
df = pd.read_excel(r'C:\Users\LAPT\Documents\JOHN\UDLA\2. CIENCIA DE DATOS Y BUSSINES INTELIGENCE\8. PROYECTO DE GRADO\BASE DE DATOS PROYECTO UDLA\data\ventas_limpias_completas.xlsx')
df['FECHA'] = pd.to_datetime(df['FECHA'])

# 2. Agrupar ventas diarias
serie = df.groupby('FECHA')['CANTIDAD'].sum().asfreq('D').fillna(0)

# 3. Prueba de estacionariedad
adf_resultado = adfuller(serie)
print("Resultado Dickey-Fuller Aumentado:")
print(f"Estadístico: {adf_resultado[0]:.4f}")
print(f"p-valor: {adf_resultado[1]:.4f}")
print("¿Serie estacionaria?", "Sí" if adf_resultado[1] < 0.05 else "No")

# 4. Preparar variable exógena
exog = df.groupby('FECHA')['PVD'].mean().asfreq('D').ffill()

# 5. Búsqueda de parámetros óptimos
mejor_aic = np.inf
mejor_bic = np.inf
mejores_params = None
mejor_modelo = None

for p in range(3):
    for d in range(2):
        for q in range(3):
            for P in range(2):
                for D in range(2):
                    for Q in range(2):
                        try:
                            modelo = sm.tsa.statespace.SARIMAX(
                                serie,
                                exog=exog,
                                order=(p, d, q),
                                seasonal_order=(P, D, Q, 7),
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                            resultado = modelo.fit(disp=False)
                            if resultado.aic < mejor_aic:
                                mejor_aic = resultado.aic
                                mejor_bic = resultado.bic
                                mejores_params = ((p, d, q), (P, D, Q, 7))
                                mejor_modelo = resultado
                        except:
                            continue

print("\n📌 Mejor combinación de parámetros encontrada:")
print(f"Parametros SARIMA: {mejores_params[0]}, Estacionales: {mejores_params[1]}")
print(f"AIC: {mejor_aic:.2f}")
print(f"BIC: {mejor_bic:.2f}")
