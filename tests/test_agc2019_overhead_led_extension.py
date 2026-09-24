import json
from pathlib import Path

from scripts.build_agc2019_overhead_led_extension import build_extension


def flatten(extension):
    return {name: spec for group in extension.values() if isinstance(group, dict)
            for name, spec in group.items() if isinstance(spec, dict) and "type" in spec}


def test_extension_adds_separate_led_state_without_interlight_geometry():
    variables = flatten(build_extension())
    assert variables["tLed"]["type"] == "state"
    assert "qLedProcessed" in variables["tLed"]["definition"]
    assert "tIntLamp" not in variables["tLed"]["definition"]
    assert variables["qLampIn"]["definition"] == "qHpsProcessed"


def test_every_led_energy_receiver_is_connected_to_its_state_balance():
    variables = flatten(build_extension())
    receivers = {
        "rLedCovIn": "tCovIn", "rLedBlScr": "tBlScr", "rLedThScr": "tThScr",
        "hLedAir": "tAir", "rLedAir": "tAir", "rParLedCan": "tCan",
        "rNirLedCan": "tCan", "rFirLedCan": "tCan", "rLedPipe": "tPipe",
        "rParLedFlr": "tFlr", "rNirLedFlr": "tFlr", "rFirLedFlr": "tFlr",
    }
    for flux, state in receivers.items():
        assert flux in variables[state]["definition"]
        assert flux in variables["tLed"]["definition"]


def test_photosynthesis_gets_par_photons_but_not_far_red_photons():
    variables = flatten(build_extension())
    assert "parLedCan" in variables["parCan"]["definition"]
    assert "ledParPhotonFlux" in variables["parLedCan"]["definition"]
    assert "ledFarRedPhotonFlux" not in variables["parCan"]["definition"]


def test_metadata_marks_seed_parameters_unvalidated():
    info = build_extension()["Info"]
    assert "uncalibrated" in info["Status"]
    assert "not AGC measurements" in info["Parameter provenance"]


def test_committed_extension_matches_builder():
    path = Path(__file__).resolve().parents[1] / "configs" / "agc2019_greenlight_overhead_led_extension_v0.json"
    assert json.loads(path.read_text()) == build_extension()
