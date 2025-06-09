# analisis_descriptivo.py

import os
import pandas as pd

# Definir la ruta base
ruta_base = os.path.dirname(os.path.dirname(__file__))
archivo = os.path.join(ruta_base, 'data', 'ventas_limpias_completas.xlsx')

# Cargar los datos
df = pd.read_excel(archivo)
df.columns = df.columns.str.strip().str.upper()

# Filtramos las columnas numéricas relevantes
variables = ['CANTIDAD', 'TOTAL']

# Análisis descriptivo básico
descriptivo = df[variables].describe().T

# Agregamos la mediana
descriptivo['mediana'] = df[variables].median()

# Guardamos el análisis como Excel
output_path = os.path.join(ruta_base, 'data', 'analisis_descriptivo.xlsx')
descriptivo.to_excel(output_path)

# Mostrar en consola
print("✅ Análisis descriptivo generado y guardado en 'analisis_descriptivo.xlsx'")
print(descriptivo)
