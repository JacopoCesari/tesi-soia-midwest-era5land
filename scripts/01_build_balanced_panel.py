"""
01_build_balanced_panel.py

Assembles the unbroken 135-county balanced panel for soybean yields (1951-2025)
from raw USDA NASS survey exports.

Selection rule:
Keep only counties with complete, non-missing survey yield for all 75 years.
No imputation is performed.
"""

from pathlib import Path
import pandas as pd

# Define paths relative to the project root
ROOT = Path(__file__).resolve().parent.parent
YIELD_EXCEL = ROOT / "data" / "raw" / "usda_nass" / "soybean_yield_479_counties_1950_2025.xlsx"
ACRES_EXCEL = ROOT / "data" / "raw" / "usda_nass" / "soybean_acres_479_counties_1950_2025.xlsx"

TARGET_CSV = ROOT / "data" / "target" / "soybean_yield_1951_2025.csv"
COUNTIES_CSV = ROOT / "data" / "auxiliary" / "counties.csv"
ACRES_CSV = ROOT / "data" / "auxiliary" / "acres_harvested_1951_2025.csv"

START_YEAR = 1951
END_YEAR = 2025
TOTAL_YEARS = END_YEAR - START_YEAR + 1


def main():
    print("Reading raw USDA NASS survey files...")
    
    # Load all survey yield observations
    df_yield = pd.read_excel(YIELD_EXCEL, sheet_name="all_years_survey", dtype={"county_fips": str})
    df_yield["county_fips"] = df_yield["county_fips"].str.zfill(5)
    
    # Filter for the 1951-2025 study period
    df_yield = df_yield[df_yield["year"].between(START_YEAR, END_YEAR)]
    
    # Retain counties with complete coverage across all 75 consecutive years
    valid_counts = df_yield.dropna(subset=["yield_bu_per_acre"]).groupby("county_fips")["year"].nunique()
    complete_counties = valid_counts[valid_counts == TOTAL_YEARS].index
    
    panel_yield = df_yield[df_yield["county_fips"].isin(complete_counties)].copy()
    panel_yield = panel_yield.sort_values(["county_fips", "year"]).reset_index(drop=True)
    
    print(f"Selected {panel_yield['county_fips'].nunique()} counties with complete records ({len(panel_yield)} rows).")

    # Extract unique county geographic attributes
    counties_meta = panel_yield[["county_fips", "state", "county"]].drop_duplicates().sort_values("county_fips")
    
    # Load and filter harvested acreage for the same balanced county-year keys
    df_acres = pd.read_excel(ACRES_EXCEL, sheet_name="all_years_survey", dtype={"county_fips": str})
    df_acres["county_fips"] = df_acres["county_fips"].str.zfill(5)
    df_acres = df_acres[df_acres["year"].between(START_YEAR, END_YEAR)]
    panel_acres = df_acres[df_acres["county_fips"].isin(complete_counties)].copy()
    panel_acres = panel_acres.sort_values(["county_fips", "year"]).reset_index(drop=True)

    # Export cleaned target and auxiliary tables
    TARGET_CSV.parent.mkdir(parents=True, exist_ok=True)
    COUNTIES_CSV.parent.mkdir(parents=True, exist_ok=True)

    panel_yield[["county_fips", "year", "yield_bu_per_acre"]].to_csv(TARGET_CSV, index=False)
    counties_meta.to_csv(COUNTIES_CSV, index=False)
    panel_acres[["county_fips", "year", "acres_harvested"]].to_csv(ACRES_CSV, index=False)
    
    print("Balanced panel successfully written to target and auxiliary files.")


if __name__ == "__main__":
    main()
