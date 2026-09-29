from scripts.agc.build_agc2019_heating_extension import build_extension


def variables():
    return {name: spec for group in build_extension().values() if isinstance(group, dict)
            for name, spec in group.items() if isinstance(spec, dict) and "type" in spec}


def test_pipe_tracking_is_capacity_bounded_and_off_gated():
    x=variables()
    assert "pipeLowActive * min(railPipePeak" in x["hBoilPipe"]["definition"]
    assert "pipeGrowActive * min(growPipePeak" in x["hBoilGroPipe"]["definition"]
    assert x["railPipePeak"]["definition"] == "180"
    assert x["growPipePeak"]["definition"] == "30"


def test_zero_source_code_is_not_a_temperature_target():
    info=build_extension()["Info"]
    assert "zero is an off/status flag" in info["Semantics"]
    assert "no source value is treated as heat power" in info["Semantics"]
