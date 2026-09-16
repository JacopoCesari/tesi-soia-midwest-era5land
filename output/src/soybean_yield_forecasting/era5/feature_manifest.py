"""Canonical 37-feature manifest for ERA5-Land acquisition and ARCO migration.

This module defines the versioned specification of the 37 original expected features,
their ECMWF metadata, ARCO availability, daily aggregation rules, physical unit conversions,
exact/proxy classifications, lifecycle status, and automated schema validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


ALLOWED_STATUSES = {
    "ARCO",
    "DERIVED_EXACT",
    "DERIVED_PROXY",
    "OPTIONAL_CDS_WISHLIST",
    "NOT_ACQUIRED_WITH_REASON",
}

ALLOWED_CLASSIFICATIONS = {"exact", "proxy", "none"}


@dataclass(frozen=True)
class FeatureDefinition:
    """Specification of an ERA5-Land weather feature."""

    canonical_name: str
    ecmwf_code: str
    description: str
    final_unit: str
    arco_source: str | None
    arco_variable: str | None
    daily_aggregation: str | None
    formula_and_dependencies: str | None
    classification: str
    status: str
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Canonical Registry of the 37 Original ERA5-Land Features
# -----------------------------------------------------------------------------

ORIGINAL_37_FEATURES: list[FeatureDefinition] = [
    # --- 1. Air Temperature Mean ---
    FeatureDefinition(
        canonical_name="air_temperature_mean",
        ecmwf_code="2t (paramId 167)",
        description="2 metre temperature daily mean",
        final_unit="°C",
        arco_source="sfc-2m-temperature",
        arco_variable="t2m",
        daily_aggregation="mean",
        formula_and_dependencies="daily_mean(t2m) - 273.15",
        classification="exact",
        status="ARCO",
        reason=None,
    ),
    # --- 2. Air Temperature Minimum ---
    FeatureDefinition(
        canonical_name="air_temperature_minimum",
        ecmwf_code="2t (paramId 167)",
        description="2 metre temperature daily minimum",
        final_unit="°C",
        arco_source="sfc-2m-temperature",
        arco_variable="t2m",
        daily_aggregation="min",
        formula_and_dependencies="daily_min(t2m) - 273.15",
        classification="exact",
        status="ARCO",
        reason=None,
    ),
    # --- 3. Air Temperature Maximum ---
    FeatureDefinition(
        canonical_name="air_temperature_maximum",
        ecmwf_code="2t (paramId 167)",
        description="2 metre temperature daily maximum",
        final_unit="°C",
        arco_source="sfc-2m-temperature",
        arco_variable="t2m",
        daily_aggregation="max",
        formula_and_dependencies="daily_max(t2m) - 273.15",
        classification="exact",
        status="ARCO",
        reason=None,
    ),
    # --- 4. Skin Temperature Maximum ---
    FeatureDefinition(
        canonical_name="skin_temperature_maximum",
        ecmwf_code="skt (paramId 235)",
        description="Skin temperature daily maximum",
        final_unit="°C",
        arco_source="sfc-skin-temperature",
        arco_variable="skt",
        daily_aggregation="max",
        formula_and_dependencies="daily_max(skt) - 273.15",
        classification="exact",
        status="ARCO",
        reason=None,
    ),
    # --- 5. Dewpoint Temperature Mean ---
    FeatureDefinition(
        canonical_name="dewpoint_temperature_mean",
        ecmwf_code="2d (paramId 168)",
        description="2 metre dewpoint temperature daily mean",
        final_unit="°C",
        arco_source="sfc-2m-temperature",
        arco_variable="d2m",
        daily_aggregation="mean",
        formula_and_dependencies="daily_mean(d2m) - 273.15",
        classification="exact",
        status="ARCO",
        reason=None,
    ),
    # --- 6. Eastward Wind Mean ---
    FeatureDefinition(
        canonical_name="eastward_wind_mean",
        ecmwf_code="10u (paramId 165)",
        description="10 metre U wind component daily mean",
        final_unit="m/s",
        arco_source="sfc-wind",
        arco_variable="u10",
        daily_aggregation="mean",
        formula_and_dependencies="daily_mean(u10)",
        classification="exact",
        status="ARCO",
        reason=None,
    ),
    # --- 7. Northward Wind Mean ---
    FeatureDefinition(
        canonical_name="northward_wind_mean",
        ecmwf_code="10v (paramId 166)",
        description="10 metre V wind component daily mean",
        final_unit="m/s",
        arco_source="sfc-wind",
        arco_variable="v10",
        daily_aggregation="mean",
        formula_and_dependencies="daily_mean(v10)",
        classification="exact",
        status="ARCO",
        reason=None,
    ),
    # --- 8. Surface Pressure ---
    FeatureDefinition(
        canonical_name="surface_pressure",
        ecmwf_code="sp (paramId 134)",
        description="Surface pressure daily mean",
        final_unit="hPa",
        arco_source="sfc-pressure-precipitation",
        arco_variable="sp",
        daily_aggregation="mean",
        formula_and_dependencies="daily_mean(sp) / 100.0",
        classification="exact",
        status="ARCO",
        reason=None,
    ),
    # --- 9. Soil Temperature Level 1 ---
    FeatureDefinition(
        canonical_name="soil_temperature_level_1",
        ecmwf_code="stl1 (paramId 139)",
        description="Soil temperature level 1 (0-7 cm) daily mean",
        final_unit="°C",
        arco_source="sfc-soil-temperature",
        arco_variable="stl1",
        daily_aggregation="mean",
        formula_and_dependencies="daily_mean(stl1) - 273.15",
        classification="exact",
        status="ARCO",
        reason=None,
    ),
    # --- 10. Soil Temperature Level 2 ---
    FeatureDefinition(
        canonical_name="soil_temperature_level_2",
        ecmwf_code="stl2 (paramId 170)",
        description="Soil temperature level 2 (7-28 cm) daily mean",
        final_unit="°C",
        arco_source="sfc-soil-temperature",
        arco_variable="stl2",
        daily_aggregation="mean",
        formula_and_dependencies="daily_mean(stl2) - 273.15",
        classification="exact",
        status="ARCO",
        reason=None,
    ),
    # --- 11. Soil Temperature Level 3 ---
    FeatureDefinition(
        canonical_name="soil_temperature_level_3",
        ecmwf_code="stl3 (paramId 183)",
        description="Soil temperature level 3 (28-100 cm) daily mean",
        final_unit="°C",
        arco_source="sfc-soil-temperature",
        arco_variable="stl3",
        daily_aggregation="mean",
        formula_and_dependencies="daily_mean(stl3) - 273.15",
        classification="exact",
        status="ARCO",
        reason=None,
    ),
    # --- 12. Soil Temperature Level 4 ---
    FeatureDefinition(
        canonical_name="soil_temperature_level_4",
        ecmwf_code="stl4 (paramId 236)",
        description="Soil temperature level 4 (100-289 cm) daily mean",
        final_unit="°C",
        arco_source="sfc-soil-temperature",
        arco_variable="stl4",
        daily_aggregation="mean",
        formula_and_dependencies="daily_mean(stl4) - 273.15",
        classification="exact",
        status="ARCO",
        reason=None,
    ),
    # --- 13. Volumetric Soil Water Layer 1 ---
    FeatureDefinition(
        canonical_name="volumetric_soil_water_layer_1",
        ecmwf_code="swvl1 (paramId 39)",
        description="Volumetric soil water layer 1 (0-7 cm) daily mean",
        final_unit="m³/m³",
        arco_source="sfc-soil-water",
        arco_variable="swvl1",
        daily_aggregation="mean",
        formula_and_dependencies="daily_mean(swvl1)",
        classification="exact",
        status="ARCO",
        reason=None,
    ),
    # --- 14. Volumetric Soil Water Layer 2 ---
    FeatureDefinition(
        canonical_name="volumetric_soil_water_layer_2",
        ecmwf_code="swvl2 (paramId 40)",
        description="Volumetric soil water layer 2 (7-28 cm) daily mean",
        final_unit="m³/m³",
        arco_source="sfc-soil-water",
        arco_variable="swvl2",
        daily_aggregation="mean",
        formula_and_dependencies="daily_mean(swvl2)",
        classification="exact",
        status="ARCO",
        reason=None,
    ),
    # --- 15. Volumetric Soil Water Layer 3 ---
    FeatureDefinition(
        canonical_name="volumetric_soil_water_layer_3",
        ecmwf_code="swvl3 (paramId 41)",
        description="Volumetric soil water layer 3 (28-100 cm) daily mean",
        final_unit="m³/m³",
        arco_source="sfc-soil-water",
        arco_variable="swvl3",
        daily_aggregation="mean",
        formula_and_dependencies="daily_mean(swvl3)",
        classification="exact",
        status="ARCO",
        reason=None,
    ),
    # --- 16. Volumetric Soil Water Layer 4 ---
    FeatureDefinition(
        canonical_name="volumetric_soil_water_layer_4",
        ecmwf_code="swvl4 (paramId 42)",
        description="Volumetric soil water layer 4 (100-289 cm) daily mean",
        final_unit="m³/m³",
        arco_source="sfc-soil-water",
        arco_variable="swvl4",
        daily_aggregation="mean",
        formula_and_dependencies="daily_mean(swvl4)",
        classification="exact",
        status="ARCO",
        reason=None,
    ),
    # --- 17. Skin Reservoir Content ---
    FeatureDefinition(
        canonical_name="skin_reservoir_content",
        ecmwf_code="src (paramId 198)",
        description="Skin reservoir content",
        final_unit="mm",
        arco_source=None,
        arco_variable=None,
        daily_aggregation="mean",
        formula_and_dependencies=None,
        classification="none",
        status="NOT_ACQUIRED_WITH_REASON",
        reason="Variable not published in ECMWF ARCO ERA5-Land Zarr stores; cannot be physically derived without prognostic land surface canopy model.",
    ),
    # --- 18. Snow Cover ---
    FeatureDefinition(
        canonical_name="snow_cover",
        ecmwf_code="snowc (paramId 243)",
        description="Snow cover fraction daily mean",
        final_unit="%",
        arco_source="sfc-snow",
        arco_variable="snowc",
        daily_aggregation="mean",
        formula_and_dependencies="daily_mean(snowc)",
        classification="exact",
        status="ARCO",
        reason=None,
    ),
    # --- 19. Snow Density ---
    FeatureDefinition(
        canonical_name="snow_density",
        ecmwf_code="rsn (paramId 33)",
        description="Snow density",
        final_unit="kg/m³",
        arco_source=None,
        arco_variable=None,
        daily_aggregation="mean",
        formula_and_dependencies=None,
        classification="none",
        status="NOT_ACQUIRED_WITH_REASON",
        reason="Variable not published in ECMWF ARCO ERA5-Land Zarr stores.",
    ),
    # --- 20. Snow Depth Water Equivalent ---
    FeatureDefinition(
        canonical_name="snow_depth_water_equivalent",
        ecmwf_code="sd (paramId 141)",
        description="Snow depth water equivalent",
        final_unit="mm",
        arco_source=None,
        arco_variable=None,
        daily_aggregation="mean",
        formula_and_dependencies=None,
        classification="none",
        status="NOT_ACQUIRED_WITH_REASON",
        reason="Variable not published in ECMWF ARCO ERA5-Land Zarr stores; physical water equivalent requires snow density (rsn), also absent. Artificial reconstruction is prohibited.",
    ),
    # --- 21. Total Precipitation ---
    FeatureDefinition(
        canonical_name="total_precipitation",
        ecmwf_code="tp (paramId 228)",
        description="Total precipitation daily cumulative sum",
        final_unit="mm",
        arco_source="sfc-pressure-precipitation",
        arco_variable="tp",
        daily_aggregation="sum",
        formula_and_dependencies="daily_sum(tp) * 1000.0",
        classification="exact",
        status="ARCO",
        reason=None,
    ),
    # --- 22. Total Evaporation ---
    FeatureDefinition(
        canonical_name="total_evaporation",
        ecmwf_code="e (paramId 182)",
        description="Total evaporation",
        final_unit="mm",
        arco_source=None,
        arco_variable=None,
        daily_aggregation="sum",
        formula_and_dependencies=None,
        classification="none",
        status="OPTIONAL_CDS_WISHLIST",
        reason="Not available in ARCO Zarr stores; retained on CDS wishlist for future acquisition if CDS retrieval queues permit.",
    ),
    # --- 23. Evaporation from the Top of Canopy ---
    FeatureDefinition(
        canonical_name="evaporation_from_the_top_of_canopy",
        ecmwf_code="evatc (paramId 260028)",
        description="Evaporation from the top of canopy",
        final_unit="mm",
        arco_source=None,
        arco_variable=None,
        daily_aggregation="sum",
        formula_and_dependencies=None,
        classification="none",
        status="NOT_ACQUIRED_WITH_REASON",
        reason="Variable not published in ECMWF ARCO ERA5-Land Zarr stores.",
    ),
    # --- 24. Potential Evaporation ---
    FeatureDefinition(
        canonical_name="potential_evaporation",
        ecmwf_code="pev (paramId 228251)",
        description="Potential evaporation",
        final_unit="mm",
        arco_source=None,
        arco_variable=None,
        daily_aggregation="sum",
        formula_and_dependencies=None,
        classification="none",
        status="NOT_ACQUIRED_WITH_REASON",
        reason="Variable not published in ECMWF ARCO ERA5-Land Zarr stores; cannot be renamed or substituted with ET0_FAO56 because IFS prognostic potential_evaporation uses a distinct formulation.",
    ),
    # --- 25. Surface Runoff ---
    FeatureDefinition(
        canonical_name="surface_runoff",
        ecmwf_code="sro (paramId 8)",
        description="Surface runoff",
        final_unit="mm",
        arco_source=None,
        arco_variable=None,
        daily_aggregation="sum",
        formula_and_dependencies=None,
        classification="none",
        status="OPTIONAL_CDS_WISHLIST",
        reason="Not available in ARCO Zarr stores; retained on CDS wishlist for future acquisition if CDS retrieval queues permit.",
    ),
    # --- 26. Sub-surface Runoff ---
    FeatureDefinition(
        canonical_name="sub_surface_runoff",
        ecmwf_code="ssro (paramId 9)",
        description="Sub-surface runoff",
        final_unit="mm",
        arco_source=None,
        arco_variable=None,
        daily_aggregation="sum",
        formula_and_dependencies=None,
        classification="none",
        status="OPTIONAL_CDS_WISHLIST",
        reason="Not available in ARCO Zarr stores; retained on CDS wishlist for future acquisition if CDS retrieval queues permit.",
    ),
    # --- 27. Snowfall ---
    FeatureDefinition(
        canonical_name="snowfall",
        ecmwf_code="sf (paramId 144)",
        description="Snowfall water equivalent",
        final_unit="mm",
        arco_source=None,
        arco_variable=None,
        daily_aggregation="sum",
        formula_and_dependencies=None,
        classification="none",
        status="NOT_ACQUIRED_WITH_REASON",
        reason="Variable not published in ECMWF ARCO ERA5-Land Zarr stores.",
    ),
    # --- 28. Snowmelt ---
    FeatureDefinition(
        canonical_name="snowmelt",
        ecmwf_code="smlt (paramId 45)",
        description="Snowmelt",
        final_unit="mm",
        arco_source=None,
        arco_variable=None,
        daily_aggregation="sum",
        formula_and_dependencies=None,
        classification="none",
        status="NOT_ACQUIRED_WITH_REASON",
        reason="Variable not published in ECMWF ARCO ERA5-Land Zarr stores; physical snowmelt requires complete surface energy balance. Artificial reconstruction is prohibited.",
    ),
    # --- 29. Surface Solar Radiation Downwards ---
    FeatureDefinition(
        canonical_name="surface_solar_radiation_downwards",
        ecmwf_code="ssrd (paramId 169)",
        description="Surface solar radiation downwards daily cumulative sum",
        final_unit="MJ/m²",
        arco_source="sfc-radiation-heat",
        arco_variable="ssrd",
        daily_aggregation="sum",
        formula_and_dependencies="daily_sum(ssrd) / 1_000_000.0",
        classification="exact",
        status="ARCO",
        reason=None,
    ),
    # --- 30. Surface Thermal Radiation Downwards ---
    FeatureDefinition(
        canonical_name="surface_thermal_radiation_downwards",
        ecmwf_code="strd (paramId 175)",
        description="Surface thermal radiation downwards daily cumulative sum",
        final_unit="MJ/m²",
        arco_source="sfc-radiation-heat",
        arco_variable="strd",
        daily_aggregation="sum",
        formula_and_dependencies="daily_sum(strd) / 1_000_000.0",
        classification="exact",
        status="ARCO",
        reason=None,
    ),
    # --- 31. Surface Net Solar Radiation ---
    FeatureDefinition(
        canonical_name="surface_net_solar_radiation",
        ecmwf_code="ssr (paramId 176)",
        description="Surface net solar radiation",
        final_unit="MJ/m²",
        arco_source=None,
        arco_variable=None,
        daily_aggregation="sum",
        formula_and_dependencies=None,
        classification="none",
        status="OPTIONAL_CDS_WISHLIST",
        reason="Not available in ARCO Zarr stores; retained on CDS wishlist for future acquisition if CDS retrieval queues permit.",
    ),
    # --- 32. Surface Net Thermal Radiation ---
    FeatureDefinition(
        canonical_name="surface_net_thermal_radiation",
        ecmwf_code="str (paramId 177)",
        description="Surface net thermal radiation",
        final_unit="MJ/m²",
        arco_source=None,
        arco_variable=None,
        daily_aggregation="sum",
        formula_and_dependencies=None,
        classification="none",
        status="NOT_ACQUIRED_WITH_REASON",
        reason="Variable not published in ECMWF ARCO ERA5-Land Zarr stores; requires surface upward thermal emission not provided in ARCO. Artificial reconstruction is prohibited.",
    ),
    # --- 33. Surface Sensible Heat Flux ---
    FeatureDefinition(
        canonical_name="surface_sensible_heat_flux",
        ecmwf_code="sshf (paramId 146)",
        description="Surface sensible heat flux",
        final_unit="MJ/m²",
        arco_source=None,
        arco_variable=None,
        daily_aggregation="sum",
        formula_and_dependencies=None,
        classification="none",
        status="NOT_ACQUIRED_WITH_REASON",
        reason="Variable not published in ECMWF ARCO ERA5-Land Zarr stores.",
    ),
    # --- 34. Surface Latent Heat Flux ---
    FeatureDefinition(
        canonical_name="surface_latent_heat_flux",
        ecmwf_code="slhf (paramId 147)",
        description="Surface latent heat flux",
        final_unit="MJ/m²",
        arco_source=None,
        arco_variable=None,
        daily_aggregation="sum",
        formula_and_dependencies=None,
        classification="none",
        status="NOT_ACQUIRED_WITH_REASON",
        reason="Variable not published in ECMWF ARCO ERA5-Land Zarr stores.",
    ),
    # --- 35. Evaporation from Bare Soil ---
    FeatureDefinition(
        canonical_name="evaporation_from_bare_soil",
        ecmwf_code="evabs (paramId 260029)",
        description="Evaporation from bare soil",
        final_unit="mm",
        arco_source=None,
        arco_variable=None,
        daily_aggregation="sum",
        formula_and_dependencies=None,
        classification="none",
        status="NOT_ACQUIRED_WITH_REASON",
        reason="Variable not published in ECMWF ARCO ERA5-Land Zarr stores. (ECMWF mapping: evabs = evaporation from bare soil).",
    ),
    # --- 36. Evaporation from Vegetation Transpiration ---
    FeatureDefinition(
        canonical_name="evaporation_from_vegetation_transpiration",
        ecmwf_code="evavt (paramId 260030)",
        description="Evaporation from vegetation transpiration",
        final_unit="mm",
        arco_source=None,
        arco_variable=None,
        daily_aggregation="sum",
        formula_and_dependencies=None,
        classification="none",
        status="NOT_ACQUIRED_WITH_REASON",
        reason="Variable not published in ECMWF ARCO ERA5-Land Zarr stores. (ECMWF mapping: evavt = evaporation from vegetation transpiration).",
    ),
    # --- 37. Evaporation from Open Water Surfaces Excluding Oceans ---
    FeatureDefinition(
        canonical_name="evaporation_from_open_water_surfaces_excluding_oceans",
        ecmwf_code="evaow (paramId 260031)",
        description="Evaporation from open water surfaces excluding oceans",
        final_unit="mm",
        arco_source=None,
        arco_variable=None,
        daily_aggregation="sum",
        formula_and_dependencies=None,
        classification="none",
        status="NOT_ACQUIRED_WITH_REASON",
        reason="Variable not published in ECMWF ARCO ERA5-Land Zarr stores. (ECMWF mapping: evaow = evaporation from open water surfaces excluding oceans).",
    ),
]


FEATURE_DICT: dict[str, FeatureDefinition] = {f.canonical_name: f for f in ORIGINAL_37_FEATURES}


# -----------------------------------------------------------------------------
# Automated Validation
# -----------------------------------------------------------------------------


def validate_feature_manifest(features: list[FeatureDefinition] | None = None) -> dict[str, Any]:
    """Validate completeness, uniqueness, and integrity of the 37-feature manifest.

    Verification requirements:
    - Exactly 37 expected features.
    - 0 missing features.
    - 0 duplicate features.
    - 0 features without valid status.
    - Explicit verification of ECMWF evaporation mappings:
        evabs -> evaporation from bare soil
        evaow -> evaporation from open water surfaces excluding oceans
        evatc -> evaporation from the top of canopy
        evavt -> evaporation from vegetation transpiration
    - Explicit verification of the 4 CDS wishlist variables:
        total_evaporation, surface_runoff, sub_surface_runoff, surface_net_solar_radiation.
    """
    items = features if features is not None else ORIGINAL_37_FEATURES
    total_count = len(items)
    if total_count != 37:
        raise ValueError(f"Feature count mismatch: expected 37, found {total_count}")

    names = [f.canonical_name for f in items]
    unique_names = set(names)
    duplicates = [name for name in unique_names if names.count(name) > 1]
    if duplicates:
        raise ValueError(f"Duplicate features found: {duplicates}")

    untyped_features = [f.canonical_name for f in items if not f.status or f.status not in ALLOWED_STATUSES]
    if untyped_features:
        raise ValueError(f"Features with missing or invalid status: {untyped_features}")

    invalid_classifications = [
        f.canonical_name for f in items if not f.classification or f.classification not in ALLOWED_CLASSIFICATIONS
    ]
    if invalid_classifications:
        raise ValueError(f"Features with invalid classification: {invalid_classifications}")

    lookup = {f.canonical_name: f for f in items}

    # Verify ECMWF evaporation mappings
    evap_checks = {
        "evaporation_from_bare_soil": "evabs",
        "evaporation_from_open_water_surfaces_excluding_oceans": "evaow",
        "evaporation_from_the_top_of_canopy": "evatc",
        "evaporation_from_vegetation_transpiration": "evavt",
    }
    for canonical, expected_code in evap_checks.items():
        if canonical not in lookup:
            raise ValueError(f"Missing evaporation feature: {canonical}")
        if expected_code not in lookup[canonical].ecmwf_code:
            raise ValueError(
                f"ECMWF code mismatch for {canonical}: expected code containing '{expected_code}', got '{lookup[canonical].ecmwf_code}'"
            )

    # Verify 4 Wishlist items
    expected_wishlist = {
        "total_evaporation",
        "surface_runoff",
        "sub_surface_runoff",
        "surface_net_solar_radiation",
    }
    for name in expected_wishlist:
        if name not in lookup:
            raise ValueError(f"Missing wishlist feature: {name}")
        if lookup[name].status != "OPTIONAL_CDS_WISHLIST":
            raise ValueError(
                f"Wishlist feature {name} has incorrect status: {lookup[name].status}, expected OPTIONAL_CDS_WISHLIST"
            )

    # Status counts
    status_counts: dict[str, int] = {}
    for f in items:
        status_counts[f.status] = status_counts.get(f.status, 0) + 1

    return {
        "total_features": total_count,
        "missing_features": 0,
        "duplicate_features": 0,
        "features_without_status": 0,
        "status_counts": status_counts,
        "validation_status": "PASS",
    }


def save_manifest_yaml(path: Path) -> None:
    """Export the validated manifest to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_data = {
        "version": "1.0",
        "expected_count": 37,
        "features": [f.to_dict() for f in ORIGINAL_37_FEATURES],
    }
    with open(path, "w", encoding="utf-8") as file:
        yaml.dump(manifest_data, file, sort_keys=False, allow_unicode=True)


def load_manifest_yaml(path: Path) -> list[FeatureDefinition]:
    """Load and validate manifest from YAML file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    features = [FeatureDefinition(**item) for item in data["features"]]
    validate_feature_manifest(features)
    return features
