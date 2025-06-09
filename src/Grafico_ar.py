import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

# Cargar los datos actualizados
file_path =r'C:\Users\LAPT\Documents\JOHN\UDLA\2. CIENCIA DE DATOS Y BUSSINES INTELIGENCE\8. PROYECTO DE GRADO\BASE DE DATOS PROYECTO UDLA\data\ventas_limpias_completas.xlsx'
df = pd.read_excel(file_path)
df['FECHA'] = pd.to_datetime(df['FECHA'])

# Agrupar por fecha y sumar las cantidades
serie = df.groupby('FECHA')['CANTIDAD'].sum().asfreq('D').fillna(0)

# Dividir en datos de entrenamiento y prueba (últimos 30 días como prueba)
train = serie[:-30]
test = serie[-30:]

# Entrenar modelo AR
model = AutoReg(train, lags=1).fit()
predictions = model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

# Crear DataFrame para comparar
comparison_df = pd.DataFrame({'Fecha': test.index, 'Real': test.values, 'Pronosticado': predictions.values})
comparison_df.set_index('Fecha', inplace=True)

# Graficar
plt.figure(figsize=(12, 6))
plt.plot(comparison_df.index, comparison_df['Real'], label='Real', marker='o')
plt.plot(comparison_df.index, comparison_df['Pronosticado'], label='Pronosticado', linestyle='--', marker='x')
plt.title('Comparación de Predicciones AR vs Valores Reales (Abril-Mayo 2025)')
plt.xlabel('Fecha')
plt.ylabel('Cantidad Vendida')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/mnt/data/comparacion_ar_real.png")
plt.show()

