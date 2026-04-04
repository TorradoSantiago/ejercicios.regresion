from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
from scipy.stats import chi2_contingency

DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "data" / "argentina_dataset_122.sav"

ISSUES = {
    1: "Unemployment",
    2: "Corruption",
    3: "Poverty and marginalization",
    4: "Healthcare",
    5: "Economic problems",
    6: "Fiscal problems",
    7: "Social problems",
    8: "Government problems",
    9: "Institutional problems",
    10: "Political culture problems",
    11: "Lack of democracy",
    12: "International issues",
    13: "Geopolitical conflicts",
    14: "Public insecurity and crime",
    15: "Drug trafficking",
    16: "Public order problems",
    17: "Territorial model of the State",
    18: "Economic model problems",
    19: "Political-institutional reforms",
    20: "Problems in the administration of justice",
    21: "Ungovernability and democratic deficit",
    22: "Human rights violations and respect for minorities",
    23: "Lack of education",
    24: "Public policy design problems",
    25: "Productivity problems",
    26: "Lack of agrarian reform",
    27: "Environmental problems",
    28: "State-related problems",
    29: "Political problems",
    30: "Pressure from economic groups",
    31: "Infrastructure problems",
    32: "Energy problems",
    33: "Bureaucracy",
    34: "The opposition",
    35: "Labor problems",
    36: "Public enterprise management model",
    37: "Agricultural sector problems",
    38: "Social movements",
    39: "Cultural problem",
    40: "National identity problems",
    41: "Populism",
    42: "Disasters",
    43: "Peace process",
    44: "Lack of independence of State institutions",
    45: "Drug addiction",
    46: "Migration",
    47: "Public services, energy, or water",
    48: "Illiteracy",
    49: "Lack of social investment",
    50: "International economy and foreign trade",
    51: "Party-related issues",
    52: "Effects of war",
    53: "Economic policy",
    54: "Electoral issues",
    55: "Lack of reforms",
    56: "Problems with the political-institutional framework",
    57: "Struggle between branches of government",
    58: "Haitian issue",
    59: "Gender violence",
    60: "Regional problems",
    61: "Economic and political dependence",
    62: "DK/NA",
}

PARTIES = {
    1: "PJ",
    2: "UCR",
    3: "UCD",
    4: "Democratic Party of Mendoza",
    6: "Intransigent Party",
    7: "FREPASO",
    8: "Socialist Party",
    13: "Neuquino People's Movement",
    14: "ARI",
    22: "Patria Grande Front",
    29: "Renewal Party of Salta",
    35: "MID",
    47: "Patria Libre",
    49: "Front for Victory",
    50: "Civic Coalition",
    53: "Nuevo Encuentro",
    54: "GEN",
    1381: "PRO",
    1382: "Communist Party",
    1386: "Patagonian Social Party",
    1390: "UNIR",
    2025: "Front of All",
    2026: "Radical Evolution",
    2400: "Renewal Front",
    2401: "Fueguino People's Movement",
    2402: "CREO",
    2403: "Recreate for Growth",
    2404: "Citizen Unity",
    2405: "PTS",
    2406: "PSOE",
    2407: "Dialogue Party",
    2408: "Labor and People's Party",
    2409: "United Republicans",
    2410: "Renewal Front of Concord",
    2411: "Tucuman for All",
    2412: "Third Way",
    2413: "Together We Are Rio Negro",
    2414: "Libertarian Party",
    2415: "Advanced Freedom",
    9999: "NA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the association between perceived issues and political party."
    )
    parser.add_argument(
        "--file-path",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the SAV file to analyze.",
    )
    return parser.parse_args()


def load_dataset(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file was not found: {file_path}")

    dataframe, _ = pyreadstat.read_sav(str(file_path))
    return dataframe


def prepare_analysis_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["PRO2", "MPOL101"]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]

    if missing_columns:
        raise KeyError(f"Required columns for the analysis are missing: {missing_columns}")

    clean_frame = dataframe[required_columns].copy()
    clean_frame["PRO2"] = pd.to_numeric(clean_frame["PRO2"], errors="coerce").astype("Int64")
    clean_frame["MPOL101"] = pd.to_numeric(clean_frame["MPOL101"], errors="coerce").astype("Int64")
    clean_frame = clean_frame.dropna(subset=["PRO2", "MPOL101"])

    clean_frame["Issue"] = clean_frame["PRO2"].map(ISSUES)
    clean_frame["Party"] = clean_frame["MPOL101"].map(PARTIES)

    clean_frame = clean_frame.dropna(subset=["Issue", "Party"])
    clean_frame = clean_frame.loc[~clean_frame["Issue"].isin({"DK/NA"})]
    clean_frame = clean_frame.loc[~clean_frame["Party"].isin({"NA"})]

    return clean_frame


def cramers_v(contingency_table: pd.DataFrame, chi2_value: float) -> float:
    observations = contingency_table.to_numpy().sum()
    if observations == 0:
        return float("nan")

    min_dimension = min(contingency_table.shape) - 1
    if min_dimension <= 0:
        return float("nan")

    return float(np.sqrt(chi2_value / (observations * min_dimension)))


def describe_dataset(dataframe: pd.DataFrame, analysis_frame: pd.DataFrame) -> None:
    print("=== Dataset summary ===")
    print(f"Original observations: {len(dataframe):,}")
    print(f"Observations used in the analysis: {len(analysis_frame):,}")
    print(f"Distinct issues: {analysis_frame['Issue'].nunique()}")
    print(f"Distinct parties: {analysis_frame['Party'].nunique()}")

    print("\nTop 5 most frequent issues:")
    print(analysis_frame["Issue"].value_counts().head(5).to_string())

    print("\nTop 5 most frequent parties:")
    print(analysis_frame["Party"].value_counts().head(5).to_string())


def run_association_test(analysis_frame: pd.DataFrame) -> None:
    contingency_table = pd.crosstab(analysis_frame["Party"], analysis_frame["Issue"])
    chi2_value, p_value, degrees_of_freedom, _ = chi2_contingency(contingency_table)
    effect_size = cramers_v(contingency_table, chi2_value)

    print("\n=== Chi-square test ===")
    print(f"Chi-square: {chi2_value:.3f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Degrees of freedom: {degrees_of_freedom}")
    print(f"Cramer's V: {effect_size:.3f}")

    if p_value < 0.05:
        interpretation = (
            "The independence hypothesis is rejected: there is evidence of an association "
            "between political party and the main perceived issue."
        )
    else:
        interpretation = (
            "There is not enough evidence to reject the independence hypothesis. "
            "The chi-square statistic does not imply correlation on its own."
        )

    print("\nInterpretation:")
    print(interpretation)

    print("\nFirst rows of the contingency table:")
    print(contingency_table.head().to_string())


def main() -> None:
    args = parse_args()
    dataframe = load_dataset(args.file_path)
    analysis_frame = prepare_analysis_frame(dataframe)
    describe_dataset(dataframe, analysis_frame)
    run_association_test(analysis_frame)


if __name__ == "__main__":
    main()
