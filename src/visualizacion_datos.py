# visualizacion_datos.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Crear la carpeta 'images' si no existe (usando ruta absoluta)
ruta_base = os.path.dirname(os.path.dirname(__file__))
ruta_images = os.path.join(ruta_base, 'images')
os.makedirs(ruta_images, exist_ok=True)

# Cargar el archivo limpio
archivo = os.path.join(ruta_base, 'data', 'ventas_limpias_completas.xlsx')
df = pd.read_excel(archivo)
df.columns = df.columns.str.strip().str.upper()  # Normalizamos nombres de columnas

# Configuración de gráficos
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (10, 6)

# Gráfico 1: Evolución mensual de los 3 productos más vendidos
productos_top = df['PRODUCTO'].value_counts().head(3).index
df_top = df[df['PRODUCTO'].isin(productos_top)]

for producto in productos_top:
    datos_producto = df_top[df_top['PRODUCTO'] == producto]
    serie = datos_producto.groupby(['AÑO', 'MES'])['CANTIDAD'].sum()
    serie.index = [f"{a}-{m:02d}" for a, m in serie.index]
    serie.sort_index(inplace=True)
    plt.plot(serie, label=producto)

plt.title('Evolución mensual de los 3 productos más vendidos')
plt.xlabel('Mes')
plt.ylabel('Cantidad vendida')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ruta_images, 'evolucion_top_productos.png'))
plt.show()

# Gráfico 2: Matriz de correlación
plt.figure()
sns.heatmap(df[['CANTIDAD', 'TOTAL']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de correlación entre variables numéricas')
plt.tight_layout()
plt.savefig(os.path.join(ruta_images, 'matriz_correlacion.png'))
plt.show()

# Gráfico 3: Dispersión CANTIDAD vs TOTAL
plt.figure()
sns.scatterplot(data=df, x='CANTIDAD', y='TOTAL')
plt.title('Dispersión entre cantidad y total de ventas')
plt.tight_layout()
plt.savefig(os.path.join(ruta_images, 'dispersion_cantidad_total.png'))
plt.show()

# Gráfico 4: Diagrama de cajas por producto (Top 3)
plt.figure()
sns.boxplot(data=df_top, x='PRODUCTO', y='CANTIDAD')
plt.title('Diagrama de caja: Cantidad por producto (Top 3)')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(ruta_images, 'boxplot_cantidad_productos.png'))
plt.show()


