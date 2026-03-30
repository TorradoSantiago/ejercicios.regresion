# ejercicios.regresion

Repositorio de ejercicios, notebooks y analisis cuantitativos con foco en regresion y lectura aplicada. En esta segunda pasada el proyecto se reorganizo para verse menos como una carpeta de practica y mas como un caso tecnico presentable.

## Que muestra este repo

- Base tecnica en regresion y pruebas de asociacion.
- Capacidad para limpiar y estructurar analisis a partir de datos reales.
- Mejoras de reproducibilidad y exportacion de resultados.
- Una capa documental pensada para portfolio o entrega academica.

## Cambios de esta iteracion

- `code/ejercitacion.py` ahora acepta parametros, genera tablas y exporta resultados.
- Se agrego un directorio `outputs/` para reportes generados.
- Se sumaron documentos de caso en `docs/`.
- El README ahora presenta el repo como proyecto analitico y no solo como ejercicio.

## Estructura principal

- `REGRESION_SIMPLE.ipynb`
- `REGRESION_SIMPLE_FINAL.ipynb`
- `TPFINAL_VOTOS_ROSARIO.ipynb`
- `code/ejercitacion.py`
- `docs/case-study.md`
- `docs/notebook-upgrade-guide.md`
- `data/BASEDATOS_ARGENTINA_122.sav`

## Ejecucion

```bash
python code/ejercitacion.py
```

Opciones utiles:

```bash
python code/ejercitacion.py --top-n 12
python code/ejercitacion.py --output-dir outputs/mi-analisis
python code/ejercitacion.py --file-path ruta/al/archivo.sav
```

## Salidas generadas

Cuando se ejecuta el script, exporta:

- `contingency_table.csv`
- `problem_frequency.csv`
- `party_frequency.csv`
- `top_pairs.csv`
- `executive_summary.md`

## Siguiente mejora recomendada

Pasar los notebooks principales a una estructura uniforme con introduccion, metodologia, resultados, limites y conclusion ejecutiva.
