import pandas as pd
from datetime import datetime

# Cargar los datos
df = pd.read_excel(r'C:\Users\LAPT\Documents\JOHN\UDLA\2. CIENCIA DE DATOS Y BUSSINES INTELIGENCE\8. PROYECTO DE GRADO\BASE DE DATOS PROYECTO UDLA\data\ventas_limpias_completas.xlsx')
df['FECHA'] = pd.to_datetime(df['FECHA'])

# Filtrar el rango de fechas
inicio = datetime(2024, 7, 1)
fin = datetime(2025, 5, 31)
ventas_periodo = df[(df['FECHA'] >= inicio) & (df['FECHA'] <= fin)]

# Calcular ventas totales en dólares
ventas_totales = ventas_periodo['TOTAL'].sum()
print(f"💰 Ventas totales entre julio 2024 y mayo 2025: ${ventas_totales:,.2f}")

#grafico de ventas julio2025-mayo2025

import matplotlib.pyplot as plt

# Agregar columna de año-mes
ventas_periodo['AÑO_MES'] = ventas_periodo['FECHA'].dt.to_period('M')

# Agrupar ventas por mes
ventas_mensuales = ventas_periodo.groupby('AÑO_MES')['TOTAL'].sum().reset_index()
ventas_mensuales['AÑO_MES'] = ventas_mensuales['AÑO_MES'].astype(str)

# Graficar
plt.figure(figsize=(10, 6))
plt.bar(ventas_mensuales['AÑO_MES'], ventas_mensuales['TOTAL'], color='skyblue')
plt.title('Ventas mensuales en dólares (Julio 2024 - Mayo 2025)')
plt.xlabel('Mes')
plt.ylabel('Ventas ($)')
plt.xticks(rotation=45)
plt.tight_layout()

# Mostrar total acumulado
total_acumulado = ventas_mensuales['TOTAL'].sum()
plt.figtext(0.99, 0.01, f'Total acumulado: ${total_acumulado:,.2f}', horizontalalignment='right')

# Guardar imagen
plt.savefig('images/ventas_mensuales_total.png')
plt.show()


# Agrupar cantidades por mes
cantidades_mensuales = ventas_periodo.groupby('AÑO_MES')['CANTIDAD'].sum().reset_index()
cantidades_mensuales['AÑO_MES'] = cantidades_mensuales['AÑO_MES'].astype(str)

# Graficar
plt.figure(figsize=(10, 6))
plt.bar(cantidades_mensuales['AÑO_MES'], cantidades_mensuales['CANTIDAD'], color='lightgreen')
plt.title('Cantidades vendidas por mes (Julio 2024 - Mayo 2025)')
plt.xlabel('Mes')
plt.ylabel('Unidades vendidas')
plt.xticks(rotation=45)
plt.tight_layout()

# Mostrar total acumulado de unidades
total_unidades = cantidades_mensuales['CANTIDAD'].sum()
plt.figtext(0.99, 0.01, f'Total unidades: {total_unidades:,.0f}', horizontalalignment='right')

# Guardar imagen
plt.savefig('images/cantidades_mensuales_total.png')
plt.show()



