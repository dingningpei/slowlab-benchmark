import importlib.util
from pathlib import Path


PATH = Path(__file__).resolve().parents[2] / "scripts" / "agc" / "audit_agc2023_development.py"
SPEC = importlib.util.spec_from_file_location("audit_agc2023_development", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_transplant_density_anomaly_is_repaired_only_on_frozen_sample_date():
    density, warning = MODULE.trusted_sample_density("2023-09-04", 1050)
    assert density == 48
    assert "conflicts" in warning


def test_other_sample_dates_must_match_workbook_density():
    assert MODULE.trusted_sample_density("2023-09-18", 48) == (48, None)
    assert MODULE.trusted_sample_density("2023-09-29", 25) == (25, None)
    assert MODULE.trusted_sample_density("2023-11-09", 20) == (20, None)


def test_nonfinite_or_text_values_are_missing():
    assert MODULE.finite_number("NaN") is None
    assert MODULE.finite_number(None) is None
    assert MODULE.finite_number("unknown") is None
    assert MODULE.finite_number("2.5") == 2.5
