# limpieza_datos.py actualizado

import pandas as pd

# Ruta completa al archivo original
archivo = r'C:\Users\LAPT\Documents\JOHN\UDLA\2. CIENCIA DE DATOS Y BUSSINES INTELIGENCE\8. PROYECTO DE GRADO\BASE DE DATOS PROYECTO UDLA\data\CRM LOMO FINO DATOS.xlsx'
df = pd.read_excel(archivo, sheet_name='CONS.')

# Exploramos columnas al inicio
print("Columnas originales:", df.columns.tolist())

# Renombramos columnas para eliminar posibles espacios ocultos
df.columns = df.columns.str.strip()

# Eliminamos solo la columna 'track pedido' si existe
if 'track pedido' in df.columns:
    df = df.drop(columns=['track pedido'])

# Seleccionamos columnas útiles, conservando IDs y fecha completa
columnas_utiles = ['ID_DETALLE', 'ID_VENTAS', 'CODIGO_CLIENTE', 'FECHA', 'PRODUCTO', 'CANTIDAD', 'TOTAL']
df = df[columnas_utiles]

# Convertimos fechas correctamente antes de eliminar nulos
df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce')

# Eliminamos nulos y registros inválidos
df = df.dropna(subset=['ID_DETALLE', 'ID_VENTAS', 'FECHA', 'PRODUCTO', 'CANTIDAD', 'TOTAL'])
df = df[df['CANTIDAD'] > 0]

# Agregamos columnas auxiliares por año y mes para análisis temporal
df['AÑO'] = df['FECHA'].dt.year
df['MES'] = df['FECHA'].dt.month

# Guardamos el archivo limpio con IDs incluidos
salida = r'C:\Users\LAPT\Documents\JOHN\UDLA\2. CIENCIA DE DATOS Y BUSSINES INTELIGENCE\8. PROYECTO DE GRADO\BASE DE DATOS PROYECTO UDLA\data\ventas_limpias_completas.xlsx'
df.to_excel(salida, index=False)

print("\n✅ Limpieza completada. Archivo guardado como 'ventas_limpias_completas.xlsx'")


