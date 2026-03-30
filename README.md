# ejercicios.regresion

Repositorio de ejercicios y proyectos de regresion con notebooks y codigo auxiliar. En esta iteracion se mejoro la parte mas reutilizable del repo para que no dependa de rutas absolutas ni de interpretaciones estadisticas equivocadas.

## Mejoras aplicadas

- Refactor de `code/ejercitacion.py` para usar rutas relativas.
- Limpieza del flujo de analisis para trabajar con `PRO2` y `MPOL101`.
- Correccion conceptual de la interpretacion del test chi-cuadrado.
- Incorporacion de `Cramer's V` como medida de intensidad de la asociacion.
- README y dependencias explicitadas.

## Contenido principal

- `REGRESION_SIMPLE.ipynb`
- `REGRESION_SIMPLE_FINAL.ipynb`
- `TPFINAL_VOTOS_ROSARIO.ipynb`
- `code/ejercitacion.py`
- `data/BASEDATOS_ARGENTINA_122.sav`

## Como ejecutar el script

```bash
python code/ejercitacion.py
```

Tambien puedes pasar una ruta distinta al dataset:

```bash
python code/ejercitacion.py --file-path ruta/al/archivo.sav
```

## Proximo paso recomendable

Pasar los notebooks principales a una estructura con introduccion, metodologia, resultados y conclusiones para que queden mejor presentados como piezas academicas o de portfolio.
