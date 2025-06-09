import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings("ignore")

# Cargar datos
ruta = r'C:\Users\LAPT\Documents\JOHN\UDLA\2. CIENCIA DE DATOS Y BUSSINES INTELIGENCE\8. PROYECTO DE GRADO\BASE DE DATOS PROYECTO UDLA\data\ventas_limpias_completas.xlsx'
df = pd.read_excel(ruta)
df['FECHA'] = pd.to_datetime(df['FECHA'])
serie = df.groupby('FECHA')['CANTIDAD'].sum().asfreq('D').fillna(0)

# Crear variable exógena (PVD)
exog = df.groupby('FECHA')['PVD'].mean().asfreq('D').ffill()

# Definir modelo con los mejores parámetros
modelo = sm.tsa.statespace.SARIMA(
    serie,
    exog=exog,
    order=(0, 0, 0),
    seasonal_order=(0, 0, 0, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)
resultado = modelo.fit(disp=False)

# Evaluación de residuos
residuos = resultado.resid

# Gráfico de residuos
plt.figure(figsize=(10, 4))
plt.plot(residuos)
plt.title("Residuos del modelo SARIMA")
plt.tight_layout()
plt.savefig("images/residuos_sarima.png")
plt.close()

# Histograma de residuos
plt.figure(figsize=(6, 4))
sns.histplot(residuos, kde=True)
plt.title("Distribución de los residuos")
plt.tight_layout()
plt.savefig("images/hist_residuos_sarima.png")
plt.close()

# Prueba Ljung-Box
ljung_box = acorr_ljungbox(residuos, lags=[10], return_df=True)
print("\nResultados de la prueba Ljung-Box:")
print(ljung_box)
