# analisis_ventas.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta base y carga del archivo
ruta_base = os.path.dirname(os.path.dirname(__file__))
archivo = os.path.join(ruta_base, 'data', 'ventas_limpias_completas.xlsx')
df = pd.read_excel(archivo)
df.columns = df.columns.str.strip().str.upper()

# Asegurarnos que la columna FECHA esté en formato datetime
df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce')
df.dropna(subset=['FECHA'], inplace=True)

# Crear una columna MES_ANIO para agrupar
df['MES_ANIO'] = df['FECHA'].dt.to_period('M').astype(str)
df['MES'] = df['FECHA'].dt.month

# 1. Evolución de ventas mensuales
ventas_mensuales = df.groupby('MES_ANIO')['TOTAL'].sum().reset_index()
plt.figure(figsize=(12, 6))
plt.plot(ventas_mensuales['MES_ANIO'], ventas_mensuales['TOTAL'], marker='o')
plt.xticks(rotation=45)
plt.title('Evolución de ventas mensuales')
plt.xlabel('Mes')
plt.ylabel('Total vendido ($)')
plt.tight_layout()
plt.savefig(os.path.join(ruta_base, 'images', 'ventas_mensuales_tendencia.png'))
plt.show()

# 2. Mapa de calor: productos vs meses
pivot_table = df.pivot_table(index='PRODUCTO', columns='MES', values='CANTIDAD', aggfunc='sum', fill_value=0)
plt.figure(figsize=(12, 10))
sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt='.0f')
plt.title('Mapa de calor: cantidad vendida por producto y mes')
plt.tight_layout()
plt.savefig(os.path.join(ruta_base, 'images', 'mapa_calor_productos_meses.png'))
plt.show()

# 3. Top 10 productos más vendidos
top_productos = df.groupby('PRODUCTO')['CANTIDAD'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
top_productos.plot(kind='barh', color='skyblue')
plt.title('Top 10 productos más vendidos')
plt.xlabel('Cantidad total vendida')
plt.tight_layout()
plt.savefig(os.path.join(ruta_base, 'images', 'top_10_productos.png'))
plt.show()

print("✅ Gráficos de análisis de ventas generados y guardados correctamente.")

