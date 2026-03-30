from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
from scipy.stats import chi2_contingency

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "BASEDATOS_ARGENTINA_122.sav"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "association-study"

PROBLEMAS = {
    1: "Desempleo",
    2: "Corrupcion",
    3: "Pobreza y marginacion",
    4: "Sanidad",
    5: "Problemas economicos",
    6: "Problemas fiscales",
    7: "Problemas sociales",
    8: "Problemas gubernamentales",
    9: "Problemas institucionales",
    10: "Problemas de cultura politica",
    11: "Falta de democracia",
    12: "Cuestiones de ambito internacional",
    13: "Conflictos geopoliticos",
    14: "Inseguridad ciudadana y delincuencia",
    15: "Narcotrafico",
    16: "Problemas de orden publico",
    17: "Modelo territorial del Estado",
    18: "Problemas del modelo economico",
    19: "Reformas politico-institucionales",
    20: "Problemas de la administracion de justicia",
    21: "Ingobernabilidad y deficit de la democracia",
    22: "Violacion de derechos humanos y respeto a minorias",
    23: "Falta de educacion",
    24: "Problemas de diseno de politica publica",
    25: "Problemas de productividad",
    26: "Falta de reforma agraria",
    27: "Problemas de medio ambiente",
    28: "Problemas estatales",
    29: "Problemas politicos",
    30: "Presion de grupos economicos",
    31: "Problemas de infraestructura",
    32: "Problemas energeticos",
    33: "Burocracia",
    34: "La oposicion",
    35: "Problemas laborales",
    36: "Modelo de administracion de empresas publicas",
    37: "Problemas del sector agropecuario",
    38: "Movimientos sociales",
    39: "Problema cultural",
    40: "Problemas de identidad nacional",
    41: "Populismo",
    42: "Catastrofes",
    43: "Proceso de paz",
    44: "Falta de independencia de organos del Estado",
    45: "Drogadiccion",
    46: "Migracion",
    47: "Servicios publicos, energia o agua",
    48: "Analfabetismo",
    49: "Falta de inversion social",
    50: "Economia internacional y comercio exterior",
    51: "Asuntos partidarios",
    52: "Efectos de la guerra",
    53: "Politica economica",
    54: "Asuntos electorales",
    55: "Falta de reformas",
    56: "Problemas con marco politico-institucional",
    57: "Lucha entre poderes del Estado",
    58: "Problema haitiano",
    59: "Violencia de genero",
    60: "Problemas regionales",
    61: "Dependencia economica y politica",
    62: "NS/NC",
}

PARTIDOS = {
    1: "PJ",
    2: "UCR",
    3: "UCD",
    4: "Partido Democratico de Mendoza",
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
    50: "Coalicion Civica",
    53: "Nuevo Encuentro",
    54: "GEN",
    1381: "PRO",
    1382: "Partido Comunista",
    1386: "Partido Social Patagonico",
    1390: "UNIR",
    2025: "Frente de Todos",
    2026: "Evolucion Radical",
    2400: "Frente Renovador",
    2401: "Movimiento Popular Fueguino",
    2402: "CREO",
    2403: "Recrear para el Crecimiento",
    2404: "Unidad Ciudadana",
    2405: "PTS",
    2406: "PSOE",
    2407: "Partido del Dialogo",
    2408: "Partido Trabajo y del Pueblo",
    2409: "Republicanos Unidos",
    2410: "Frente Renovador de la Concordia",
    2411: "Tucuman para Todos",
    2412: "Tercera Via",
    2413: "Juntos Somos Rio Negro",
    2414: "Partido Libertario",
    2415: "Libertad Avanzada",
    9999: "NC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analiza la asociacion entre problemas percibidos y partido politico."
    )
    parser.add_argument(
        "--file-path",
        type=Path,
        default=DEFAULT_DATASET,
        help="Ruta al archivo SAV a analizar.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directorio donde se exportaran tablas y resumenes.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Cantidad de filas a mostrar y exportar en tablas resumidas.",
    )
    return parser.parse_args()


def load_dataset(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontro el archivo de datos: {file_path}")

    dataframe, _ = pyreadstat.read_sav(str(file_path))
    return dataframe


def prepare_analysis_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["PRO2", "MPOL101"]
    missing = [column for column in required_columns if column not in dataframe.columns]

    if missing:
        raise KeyError(f"Faltan columnas requeridas para el analisis: {missing}")

    clean = dataframe[required_columns].copy()
    clean["PRO2"] = pd.to_numeric(clean["PRO2"], errors="coerce").astype("Int64")
    clean["MPOL101"] = pd.to_numeric(clean["MPOL101"], errors="coerce").astype("Int64")
    clean = clean.dropna(subset=["PRO2", "MPOL101"])

    clean["Problema"] = clean["PRO2"].map(PROBLEMAS)
    clean["Partido"] = clean["MPOL101"].map(PARTIDOS)

    clean = clean.dropna(subset=["Problema", "Partido"])
    clean = clean.loc[~clean["Problema"].isin({"NS/NC"})]
    clean = clean.loc[~clean["Partido"].isin({"NC"})]

    return clean


def cramers_v(contingency_table: pd.DataFrame, chi2_value: float) -> float:
    observations = contingency_table.to_numpy().sum()
    if observations == 0:
        return float("nan")

    min_dimension = min(contingency_table.shape) - 1
    if min_dimension <= 0:
        return float("nan")

    return float(np.sqrt(chi2_value / (observations * min_dimension)))


def build_frequency_tables(
    analysis_frame: pd.DataFrame, top_n: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    problem_frequency = (
        analysis_frame["Problema"]
        .value_counts()
        .rename_axis("problema")
        .reset_index(name="frecuencia")
        .head(top_n)
    )
    party_frequency = (
        analysis_frame["Partido"]
        .value_counts()
        .rename_axis("partido")
        .reset_index(name="frecuencia")
        .head(top_n)
    )
    top_pairs = (
        analysis_frame.groupby(["Partido", "Problema"])
        .size()
        .reset_index(name="frecuencia")
        .sort_values("frecuencia", ascending=False)
        .head(top_n)
    )

    return problem_frequency, party_frequency, top_pairs


def run_association_test(analysis_frame: pd.DataFrame) -> dict[str, object]:
    contingency_table = pd.crosstab(analysis_frame["Partido"], analysis_frame["Problema"])
    chi2_value, p_value, degrees_of_freedom, _ = chi2_contingency(contingency_table)
    effect_size = cramers_v(contingency_table, chi2_value)

    if p_value < 0.05:
        interpretation = (
            "Se rechaza la hipotesis de independencia: hay evidencia de asociacion "
            "entre partido politico y problema principal percibido."
        )
    else:
        interpretation = (
            "No hay evidencia suficiente para rechazar la hipotesis de independencia. "
            "El estadistico chi-cuadrado no implica correlacion por si mismo."
        )

    return {
        "contingency_table": contingency_table,
        "chi2": float(chi2_value),
        "p_value": float(p_value),
        "degrees_of_freedom": int(degrees_of_freedom),
        "cramers_v": float(effect_size),
        "interpretation": interpretation,
    }


def print_console_summary(
    dataframe: pd.DataFrame,
    analysis_frame: pd.DataFrame,
    problem_frequency: pd.DataFrame,
    party_frequency: pd.DataFrame,
    top_pairs: pd.DataFrame,
    stats: dict[str, object],
) -> None:
    print("=== Resumen del dataset ===")
    print(f"Observaciones originales: {len(dataframe):,}")
    print(f"Observaciones utiles para el analisis: {len(analysis_frame):,}")
    print(f"Problemas distintos: {analysis_frame['Problema'].nunique()}")
    print(f"Partidos distintos: {analysis_frame['Partido'].nunique()}")

    print("\nTop problemas:")
    print(problem_frequency.to_string(index=False))

    print("\nTop partidos:")
    print(party_frequency.to_string(index=False))

    print("\nCruces mas frecuentes:")
    print(top_pairs.to_string(index=False))

    print("\n=== Test chi-cuadrado ===")
    print(f"Chi-cuadrado: {stats['chi2']:.3f}")
    print(f"P-value: {stats['p_value']:.6f}")
    print(f"Grados de libertad: {stats['degrees_of_freedom']}")
    print(f"Cramer's V: {stats['cramers_v']:.3f}")

    print("\nInterpretacion:")
    print(stats["interpretation"])

    print("\nPrimeras filas de la tabla de contingencia:")
    print(stats["contingency_table"].head().to_string())


def build_executive_summary(
    dataframe: pd.DataFrame,
    analysis_frame: pd.DataFrame,
    problem_frequency: pd.DataFrame,
    party_frequency: pd.DataFrame,
    top_pairs: pd.DataFrame,
    stats: dict[str, object],
) -> str:
    top_problem = problem_frequency.iloc[0]
    top_party = party_frequency.iloc[0]
    top_pair = top_pairs.iloc[0]

    return f"""# Resumen ejecutivo

## Contexto

- Observaciones originales: {len(dataframe):,}
- Observaciones utiles para el analisis: {len(analysis_frame):,}
- Problemas distintos: {analysis_frame['Problema'].nunique()}
- Partidos distintos: {analysis_frame['Partido'].nunique()}

## Hallazgos descriptivos

- Problema mas frecuente: {top_problem['problema']} ({top_problem['frecuencia']} casos)
- Partido mas frecuente: {top_party['partido']} ({top_party['frecuencia']} casos)
- Cruce mas frecuente: {top_pair['Partido']} / {top_pair['Problema']} ({top_pair['frecuencia']} casos)

## Resultado estadistico

- Chi-cuadrado: {stats['chi2']:.3f}
- P-value: {stats['p_value']:.6f}
- Grados de libertad: {stats['degrees_of_freedom']}
- Cramer's V: {stats['cramers_v']:.3f}

## Interpretacion

{stats['interpretation']}
"""


def write_outputs(
    output_dir: Path,
    problem_frequency: pd.DataFrame,
    party_frequency: pd.DataFrame,
    top_pairs: pd.DataFrame,
    stats: dict[str, object],
    executive_summary: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    problem_frequency.to_csv(output_dir / "problem_frequency.csv", index=False)
    party_frequency.to_csv(output_dir / "party_frequency.csv", index=False)
    top_pairs.to_csv(output_dir / "top_pairs.csv", index=False)
    stats["contingency_table"].to_csv(output_dir / "contingency_table.csv")
    (output_dir / "executive_summary.md").write_text(
        executive_summary,
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    dataframe = load_dataset(args.file_path)
    analysis_frame = prepare_analysis_frame(dataframe)
    problem_frequency, party_frequency, top_pairs = build_frequency_tables(
        analysis_frame, args.top_n
    )
    stats = run_association_test(analysis_frame)

    print_console_summary(
        dataframe,
        analysis_frame,
        problem_frequency,
        party_frequency,
        top_pairs,
        stats,
    )

    executive_summary = build_executive_summary(
        dataframe,
        analysis_frame,
        problem_frequency,
        party_frequency,
        top_pairs,
        stats,
    )
    write_outputs(
        args.output_dir,
        problem_frequency,
        party_frequency,
        top_pairs,
        stats,
        executive_summary,
    )

    print(f"\nArchivos exportados en: {args.output_dir}")


if __name__ == "__main__":
    main()
