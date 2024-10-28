# Importar librerías
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pyreadstat

# Ruta al archivo .sav
file_path = r'C:\Users\Santi\Desktop\Ejercitación_regresión\data\BASEDATOS_ARGENTINA_122.sav'

# Leer los datos
df, meta = pyreadstat.read_sav(file_path)

#primeras filas del DataFrame
print(df.head())

# nombres de las columnas disponibles
print(df.columns)

# Guarda los nombres de las columnas en una lista
column_names = df.columns.tolist()
print(column_names)

print(df.describe())  # Estadísticas descriptivas para entender la distribución de los datos

#df_diputados, meta = pyreadstat.read_sav(file_path)

#df_diputados = df_diputados.loc[~((df_diputados['ID101'] == 98) | (df_diputados['ID101'] == 99) | 
 #                                 (df_diputados['PRO102'] == 98) | (df_diputados['PRO112'] == 99))]

# Lista de columnas de interés
columnas_interes = ['PRO2', 'MPOL101', 'MPOL102', 'MPOL103']

# Verificar que estas columnas existen en el DataFrame limpio
columnas_existentes = [col for col in columnas_interes if col in df.columns]

# Mostrar los valores de estas columnas si existen
if columnas_existentes:
    print("Mostrando las primeras filas de las columnas seleccionadas:")
    print(df[columnas_existentes].head())
else:
    print("Algunas o todas las columnas seleccionadas no existen en el DataFrame.")

# Opcionalmente, puedes también mostrar un resumen estadístico o la cantidad de valores únicos
if columnas_existentes:
    print("\nResumen estadístico de las columnas seleccionadas:")
    print(df[columnas_existentes].describe(include='all'))  # 'include=all' muestra resumen para todos los tipos de datos

    print("\nConteo de valores únicos por columna:")
    for col in columnas_existentes:
        print(f"{col}: {df[col].nunique()} valores únicos")

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Mapeo de problemas y partidos políticos según los códigos proporcionados
problemas = {
    1: "Desempleo",
    2: "Corrupción",
    3: "Pobreza, Marginación",
    4: "Sanidad",
    5: "Problemas Económicos",
    6: "Problemas Fiscales",
    7: "Problemas Sociales",
    8: "Problemas Gubernamentales",
    9: "Problemas Institucionales",
    10: "Problemas de Cultura Política",
    11: "Falta de Democracia",
    12: "Cuestiones de Ámbito Internacional",
    13: "Conflictos Geopolíticos",
    14: "Inseguridad Ciudadana y Delincuencia",
    15: "Narcotráfico",
    16: "Problemas de Orden Público",
    17: "Modelo Territorial del Estado",
    18: "Problemas del Modelo Económico",
    19: "Reformas Político-Institucionales",
    20: "Problemas de la Administración de Justicia",
    21: "Ingobernabilidad y Déficit de la Democracia",
    22: "Violación de Derechos Humanos y Respeto a Minorías",
    23: "Falta de Educación",
    24: "Problemas de Diseño de Política Pública",
    25: "Problemas de Productividad",
    26: "Falta de Reforma Agraria",
    27: "Problemas Medio Ambiente",
    28: "Problemas Estatales",
    29: "Problemas Políticos",
    30: "Presión Grupos Económicos",
    31: "Problemas de Infraestructura",
    32: "Problemas Energéticos",
    33: "Burocracia",
    34: "La Oposición (Intransigencia)",
    35: "Problemas Laborales",
    36: "Modelo Admon Empresas Públicas",
    37: "Problemas del Sector Agropecuario",
    38: "Movimientos Sociales",
    39: "Problema Cultural",
    40: "Problemas de Identidad Nacional",
    41: "Populismo",
    42: "Catástrofes",
    43: "Proceso de Paz",
    44: "Falta Independencia Órganos del Estado",
    45: "Drogadicción",
    46: "Migración",
    47: "Servicios Públicos, Falta Energía, Agua",
    48: "Analfabetismo",
    49: "Falta de Inversión Social",
    50: "Economía Internacional y Comercio Exterior",
    51: "Asuntos Partidarios",
    52: "Efectos de la Guerra",
    53: "Política Económica",
    54: "Asuntos Electorales",
    55: "Falta de Reformas",
    56: "Problemas con Marco Político-Institucional",
    57: "Lucha entre Poderes del Estado",
    58: "Problema Haitiano",
    59: "Violencia de Género",
    60: "Problemas Regionales",
    61: "Dependencia Económica y Política",
    62: "NS/NC"  # Asumiendo NS/NC como "No Sabe/No Contesta"
}

partidos = {
    1: "PJ Argentina",
    2: "UCR",
    3: "UCD",
    4: "Partido Democrático de Mendoza",
    6: "Partido Intransigente",
    7: "FREPASO",
    8: "Partido Socialista",
    13: "Movimiento Popular Neuquino",
    14: "ARI",
    22: "Frente Patria Grande",
    29: "Partido Renovador de Salta",
    35: "MID",
    47: "Patria Libre",
    49: "Frente para la Victoria",
    50: "Coalición Cívica",
    53: "Nuevo Encuentro",
    54: "GEN",
    1381: "PRO",
    1382: "Partido Comunista",
    1386: "Partido Social Patagónico",
    1390: "UNIR",
    2025: "Frente de Todos",
    2026: "Evolución Radical",
    2400: "Frente Renovador",
    2401: "Movimiento Popular Fueguino",
    2402: "CREO",
    2403: "Recrear para el Crecimiento",
    2404: "Unidad Ciudadana",
    2405: "PTS",
    2406: "PSOE",
    2407: "Partido del Diálogo",
    2408: "Partido Trabajo y del Pueblo",
    2409: "Republicanos Unidos",
    2410: "Frente Renovador de la Concordia",
    2411: "Tucumán para Todos",
    2412: "Tercera Vía",
    2413: "Juntos Somos Río Negro",
    2414: "Partido Libertario",
    2415: "Libertad Avanzada",
    9999: "NC"  # Asumiendo NC como "No Contesta"
}

# Mapeo en el DataFrame
df['Problema'] = df['PRO2'].map(problemas)
df['Partido'] = df['MPOL101'].map(partidos)  # Asumiendo que quieres usar MPOL101

# Crear tabla de contingencia
tabla_contingencia = pd.crosstab(df['Partido'], df['Problema'])

# Realizar la prueba chi-cuadrado
chi2, p_value, dof, expected = chi2_contingency(tabla_contingencia)

print("Chi-squared:", chi2)
print("P-value:", p_value)

# Si necesitas ver la tabla de contingencia
print(tabla_contingencia)
