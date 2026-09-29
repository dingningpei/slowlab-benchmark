import json
from pathlib import Path

import pytest

from scripts.agc.build_agc2019_floor_normalized_lighting_extension import AREA_SCALE, build


def test_floor_basis_is_fixed_from_published_areas(tmp_path):
    base = Path("configs/agc/agc2019_greenlight_overhead_led_extension_v0.json")
    out = tmp_path / "extension.json"
    data = build(base, out)
    assert AREA_SCALE == pytest.approx(0.8)
    inputs = data["AGC observed lighting inputs and shortwave exchanges"]
    assert inputs["qLampIn"]["definition"] == "agcGrowingToFloorArea*qHpsProcessed"
    assert inputs["rParGhLed"]["definition"].startswith("agcGrowingToFloorArea")
    assert inputs["parLedCan"]["definition"].startswith("agcGrowingToFloorArea")
    assert "agcGrowingToFloorArea*qLedProcessed" in data["AGC overhead LED state and receiver overrides"]["tLed"]["definition"]
    assert json.loads(out.read_text()) == data
