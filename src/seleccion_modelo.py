# seleccion_modelo.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# Asegurarse de que exista la carpeta 'images'
os.makedirs('images', exist_ok=True)

# 1. Cargar los datos limpios
df = pd.read_excel(r'C:\Users\LAPT\Documents\JOHN\UDLA\2. CIENCIA DE DATOS Y BUSSINES INTELIGENCE\8. PROYECTO DE GRADO\BASE DE DATOS PROYECTO UDLA\data\ventas_limpias_completas.xlsx')
df['FECHA'] = pd.to_datetime(df['FECHA'])
serie = df.groupby('FECHA')['CANTIDAD'].sum().asfreq('D').fillna(0)

# 2. Verificar estacionariedad con ADF
adf_resultado = adfuller(serie)
print("Resultado ADF:")
print(f"Estadístico: {adf_resultado[0]:.4f}")
print(f"p-valor: {adf_resultado[1]:.4f}")
print("¿Serie estacionaria?", "Sí" if adf_resultado[1] < 0.05 else "No")

# 3. Descomposición estacional
descomposicion = seasonal_decompose(serie, model='additive')
descomposicion.plot()
plt.tight_layout()
plt.savefig('images/descomposicion_temporal.png')
plt.close()

# 4. Diferenciación si no es estacionaria
serie_diff = serie.diff().dropna()

# 5. Grid search para SARIMA
mejor_aic = np.inf
mejor_bic = np.inf
mejor_modelo = None
mejores_params = None

for p in range(3):
    for d in range(2):
        for q in range(3):
            for P in range(2):
                for D in range(2):
                    for Q in range(2):
                        try:
                            modelo = sm.tsa.statespace.SARIMAX(
                                serie,
                                order=(p, d, q),
                                seasonal_order=(P, D, Q, 7),
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                            resultado = modelo.fit(disp=False)
                            if resultado.aic < mejor_aic:
                                mejor_aic = resultado.aic
                                mejor_bic = resultado.bic
                                mejor_modelo = resultado
                                mejores_params = ((p, d, q), (P, D, Q, 7))
                        except:
                            continue

print(f"📌 Mejor modelo SARIMA: {mejores_params}")
print(f"AIC: {mejor_aic:.2f}")
print(f"BIC: {mejor_bic:.2f}")

# 6. SARIMAX con variable exógena (PVD)
exog = df.groupby('FECHA')['PVD'].mean().asfreq('D').ffill()

modelo_sarimax = sm.tsa.statespace.SARIMAX(
    serie,
    exog=exog,
    order=mejores_params[0],
    seasonal_order=mejores_params[1],
    enforce_stationarity=False,
    enforce_invertibility=False
)

resultado_sarimax = modelo_sarimax.fit(disp=False)

print("\n📌 Modelo SARIMAX con PVD como exógena")
print(f"AIC: {resultado_sarimax.aic:.2f}")
print(f"BIC: {resultado_sarimax.bic:.2f}")

# 7. Comparación entre SARIMA y SARIMAX
if resultado_sarimax.aic < mejor_aic:
    print("✅ El modelo SARIMAX mejora el AIC respecto al modelo SARIMA.")
else:
    print("⚠️ El modelo SARIMAX no mejora el AIC respecto al modelo SARIMA.")





