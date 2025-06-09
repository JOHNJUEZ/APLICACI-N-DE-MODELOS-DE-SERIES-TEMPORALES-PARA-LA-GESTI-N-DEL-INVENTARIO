import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product

# Cargar datos
file_path = r'C:\Users\LAPT\Documents\JOHN\UDLA\2. CIENCIA DE DATOS Y BUSSINES INTELIGENCE\8. PROYECTO DE GRADO\BASE DE DATOS PROYECTO UDLA\data\ventas_limpias_completas.xlsx'
df = pd.read_excel(file_path)
df['FECHA'] = pd.to_datetime(df['FECHA'])
df = df.groupby('FECHA')['CANTIDAD'].sum()
df = df.resample('W').sum()

# División de datos
train = df[:-10]

# Rango de parámetros
p = d = q = range(0, 3)
P = D = Q = range(0, 2)
s = 7  # Periodo estacional semanal

# Combinaciones de parámetros
param_combinations = list(product(p, d, q, P, D, Q))

# Búsqueda del mejor modelo
best_aic = np.inf
best_order = None
best_seasonal_order = None

for order in product(p, d, q):
    for seasonal_order in product(P, D, Q):
        try:
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order + (s,))
            results = model.fit(disp=False)
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = order
                best_seasonal_order = seasonal_order
        except:
            continue

print(f"Mejor modelo: SARIMA{best_order}x{best_seasonal_order + (s,)} con AIC={best_aic}")
