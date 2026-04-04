# ejercicios.regresion

Regression exercises and analytical materials focused on methodological care, statistical interpretation, and reusable structure.

Start with:
- `BRIEF.md`
- `docs/case-study.md`

## Improvements applied

- The main script now uses relative paths.
- The analytical flow was cleaned around `PRO2` and `MPOL101`.
- The chi-square interpretation was corrected.
- `Cramer's V` was added as an association-strength measure.
- The repository now exposes a clearer script and documentation layer.

## Main contents

- `simple_regression.ipynb`
- `simple_regression_final.ipynb`
- `rosario_voting_analysis.ipynb`
- `code/exercise_analysis.py`
- `data/argentina_dataset_122.sav`

## How to run the script

```bash
python code/exercise_analysis.py
```

You can also pass a different dataset path:

```bash
python code/exercise_analysis.py --file-path path/to/dataset.sav
```

## Recommended next step

Turn the strongest notebook into a polished case with a clean introduction, methodology, results, and conclusion section.
