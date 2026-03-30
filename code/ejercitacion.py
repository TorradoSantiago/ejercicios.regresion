from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
from scipy.stats import chi2_contingency

DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "data" / "BASEDATOS_ARGENTINA_122.sav"

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


def describe_dataset(dataframe: pd.DataFrame, analysis_frame: pd.DataFrame) -> None:
    print("=== Resumen del dataset ===")
    print(f"Observaciones originales: {len(dataframe):,}")
    print(f"Observaciones utiles para el analisis: {len(analysis_frame):,}")
    print(f"Problemas distintos: {analysis_frame['Problema'].nunique()}")
    print(f"Partidos distintos: {analysis_frame['Partido'].nunique()}")

    print("\nTop 5 problemas mas frecuentes:")
    print(analysis_frame["Problema"].value_counts().head(5).to_string())

    print("\nTop 5 partidos mas frecuentes:")
    print(analysis_frame["Partido"].value_counts().head(5).to_string())


def run_association_test(analysis_frame: pd.DataFrame) -> None:
    contingency_table = pd.crosstab(analysis_frame["Partido"], analysis_frame["Problema"])
    chi2_value, p_value, degrees_of_freedom, _ = chi2_contingency(contingency_table)
    effect_size = cramers_v(contingency_table, chi2_value)

    print("\n=== Test chi-cuadrado ===")
    print(f"Chi-cuadrado: {chi2_value:.3f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Grados de libertad: {degrees_of_freedom}")
    print(f"Cramer's V: {effect_size:.3f}")

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

    print("\nInterpretacion:")
    print(interpretation)

    print("\nPrimeras filas de la tabla de contingencia:")
    print(contingency_table.head().to_string())


def main() -> None:
    args = parse_args()
    dataframe = load_dataset(args.file_path)
    analysis_frame = prepare_analysis_frame(dataframe)
    describe_dataset(dataframe, analysis_frame)
    run_association_test(analysis_frame)


if __name__ == "__main__":
    main()
