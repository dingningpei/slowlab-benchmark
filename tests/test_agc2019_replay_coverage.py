import pytest

from scripts.audit_agc2019_replay_coverage import classify, longest_consecutive_run


def complete_day():
    weather = []
    climate = []
    for index in range(289):
        time = 43815 + index / 288
        weather.append((time, {"Tout": "8", "Rhout": "80", "Windsp": "2",
                               "Pyrgeo": "-70", "Iglob": "0"}))
        climate.append((time, {
            "VentLee": "0", "Ventwind": "0", "EnScr": "70", "BlackScr": "0",
            "AssimLight": "0", "PipeLow": "0", "PipeGrow": "0",
            "co2_dos": "0", "int_blue_vip": "NaN", "int_red_vip": "NaN",
            "int_farred_vip": "NaN", "int_white_vip": "NaN",
            "Tair": "20", "Rhair": "80", "CO2air": "500",
        }))
    return weather, climate


def test_missing_led_is_only_structural_when_hps_is_off():
    weather, climate = complete_day()
    assert classify(weather, climate)["input_eligible"]
    climate[10][1]["AssimLight"] = "100"
    result = classify(weather, climate)
    assert not result["input_eligible"]
    assert result["led_unknown_while_hps_on_rows"] == 1
    assert result["led_unknown_while_hps_on_max_consecutive_rows"] == 1
    assert result["led_unknown_processing_energy_upper_kwh_m2"] == pytest.approx(61.52 / 12000)
    for field in ("int_blue_vip", "int_red_vip", "int_farred_vip", "int_white_vip"):
        climate[10][1][field] = "0"
    assert classify(weather, climate)["input_eligible"]
    assert classify(weather, climate)["led_unknown_processing_energy_upper_kwh_m2"] == 0


def test_outcome_missingness_is_reported_but_not_used_to_select_input_days():
    weather, climate = complete_day()
    climate[20][1]["CO2air"] = "NaN"
    result = classify(weather, climate)
    assert result["input_eligible"]
    assert result["climate_observation_invalid_rows_not_used_for_input_eligibility"] == 1
    weather.pop()
    assert not classify(weather, climate)["input_eligible"]


def test_longest_complete_run_does_not_bridge_missing_days():
    assert longest_consecutive_run([
        "2020-01-26", "2020-03-05", "2020-03-06", "2020-03-07", "2020-03-08",
        "2020-03-14",
    ]) == 4
