# Importar librerías
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pyreadstat

# Ruta al archivo .sav
file_path = r'C:\Users\Santi\Desktop\Ejercitación_regresión\data\BASEDATOS_ARGENTINA_122.sav'

# Leer los datos
df, meta = pyreadstat.read_sav(file_path)

# Mostrar las primeras filas del DataFrame
print(df.head())

# Mostrar los nombres de las columnas disponibles
print(df.columns)

df_diputados, meta = pyreadstat.read_sav(file_path)

df_diputados = df_diputados.loc[~((df_diputados['ID101'] == 98) | (df_diputados['ID101'] == 99) | 
                                   (df_diputados['PRO102'] == 98) | (df_diputados['PRO112'] == 99))]