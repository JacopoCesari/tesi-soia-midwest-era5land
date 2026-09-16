"""Test completeness, uniqueness, integrity and ECMWF mappings of the 37-feature manifest."""

from pathlib import Path

import pytest
from soybean_yield_forecasting.era5.feature_manifest import (
    FEATURE_DICT,
    ORIGINAL_37_FEATURES,
    FeatureDefinition,
    load_manifest_yaml,
    validate_feature_manifest,
)


def test_feature_manifest_completeness_and_uniqueness():
    """Verify exactly 37 features, 0 missing, 0 duplicate, 0 unclassified."""
    report = validate_feature_manifest()
    assert report["total_features"] == 37
    assert report["missing_features"] == 0
    assert report["duplicate_features"] == 0
    assert report["features_without_status"] == 0
    assert report["validation_status"] == "PASS"

    assert report["status_counts"]["ARCO"] == 20
    assert report["status_counts"]["OPTIONAL_CDS_WISHLIST"] == 4
    assert report["status_counts"]["NOT_ACQUIRED_WITH_REASON"] == 13
    assert sum(report["status_counts"].values()) == 37


def test_evaporation_ecmwf_mappings():
    """Verify strict ECMWF mappings for evaporation variables."""
    assert "evabs" in FEATURE_DICT["evaporation_from_bare_soil"].ecmwf_code
    assert "evaow" in FEATURE_DICT["evaporation_from_open_water_surfaces_excluding_oceans"].ecmwf_code
    assert "evatc" in FEATURE_DICT["evaporation_from_the_top_of_canopy"].ecmwf_code
    assert "evavt" in FEATURE_DICT["evaporation_from_vegetation_transpiration"].ecmwf_code


def test_wishlist_features_unsolicited_by_default():
    """Ensure wishlist variables are categorized as OPTIONAL_CDS_WISHLIST."""
    wishlist = [
        "total_evaporation",
        "surface_runoff",
        "sub_surface_runoff",
        "surface_net_solar_radiation",
    ]
    for name in wishlist:
        assert FEATURE_DICT[name].status == "OPTIONAL_CDS_WISHLIST"
        assert FEATURE_DICT[name].arco_source is None


def test_yaml_manifest_serialization(project_root):
    """Verify loading from YAML preserves integrity and validation."""
    yaml_path = project_root / "configs/feature_manifest_37.yaml"
    assert yaml_path.exists()
    features = load_manifest_yaml(yaml_path)
    assert len(features) == 37
    assert all(isinstance(f, FeatureDefinition) for f in features)
    report = validate_feature_manifest(features)
    assert report["validation_status"] == "PASS"


def test_manifest_validation_catches_invalid_inputs():
    """Verify validator catches incorrect counts, duplicates, or missing status."""
    # Test duplicate
    with pytest.raises(ValueError, match="Duplicate features"):
        validate_feature_manifest(ORIGINAL_37_FEATURES[:36] + [ORIGINAL_37_FEATURES[0]])

    # Test count mismatch
    with pytest.raises(ValueError, match="Feature count mismatch"):
        validate_feature_manifest(ORIGINAL_37_FEATURES[:36])

    # Test invalid status
    bad_feature = FeatureDefinition(
        canonical_name=ORIGINAL_37_FEATURES[0].canonical_name,
        ecmwf_code=ORIGINAL_37_FEATURES[0].ecmwf_code,
        description=ORIGINAL_37_FEATURES[0].description,
        final_unit=ORIGINAL_37_FEATURES[0].final_unit,
        arco_source=ORIGINAL_37_FEATURES[0].arco_source,
        arco_variable=ORIGINAL_37_FEATURES[0].arco_variable,
        daily_aggregation=ORIGINAL_37_FEATURES[0].daily_aggregation,
        formula_and_dependencies=ORIGINAL_37_FEATURES[0].formula_and_dependencies,
        classification=ORIGINAL_37_FEATURES[0].classification,
        status="INVALID_STATUS",
    )
    with pytest.raises(ValueError, match="missing or invalid status"):
        validate_feature_manifest([bad_feature] + ORIGINAL_37_FEATURES[1:])
