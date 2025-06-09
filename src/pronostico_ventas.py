import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
import matplotlib.pyplot as plt
from datetime import timedelta
from math import ceil
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Cargar datos
file_path = r'C:\Users\LAPT\Documents\JOHN\UDLA\2. CIENCIA DE DATOS Y BUSSINES INTELIGENCE\8. PROYECTO DE GRADO\BASE DE DATOS PROYECTO UDLA\data\ventas_limpias_completas.xlsx'
df = pd.read_excel(file_path)
df['FECHA'] = pd.to_datetime(df['FECHA'])
df = df.groupby('FECHA')['CANTIDAD'].sum().asfreq('D').fillna(0)

# División de datos
train = df[:-30]

# Rango de parámetros
p = d = q = range(0, 3)
P = D = Q = range(0, 2)
s = 7  # Periodo estacional semanal

# Búsqueda del mejor modelo SARIMA
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

# Ajustar el mejor modelo
best_model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order + (s,))
best_results = best_model.fit(disp=False)

# Pronóstico
forecast = best_results.forecast(steps=30)
forecast.index = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=30, freq='D')
forecast_df = forecast.reset_index()
forecast_df.columns = ['Fecha', 'Inventario Óptimo Estimado']

import ace_tools as tools; tools.display_dataframe_to_user(name="Pronóstico de Inventario Diario", dataframe=forecast_df)
