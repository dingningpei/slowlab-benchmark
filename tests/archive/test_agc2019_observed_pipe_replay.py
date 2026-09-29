import json
from pathlib import Path


def test_observed_pipe_replay_overrides_all_pipe_heat_paths():
    path = Path(__file__).parents[2] / "configs" / "agc" / "agc2019_greenlight_observed_pipe_replay_v0.json"
    document = json.loads(path.read_text())
    variables = {name: value for section, values in document.items()
                 if section != "Info" for name, value in values.items()}
    rail = {"rPipeCovIn", "rPipeSky", "rPipeThScr", "rPipeBlScr", "rPipeFlr",
            "rPipeCan", "hPipeAir", "rLampPipe", "rIntLampPipe", "rLedPipe"}
    assert rail <= variables.keys()
    assert all("tPipeReplay" in variables[name]["definition"] for name in rail)
    assert "pipeLowObserved" in variables["tPipeReplay"]["definition"]
    assert "pipeGrowObserved" in variables["tGroPipeReplay"]["definition"]
    assert "rLedPipe" in variables["hObservedRailPipeDemand"]["definition"]


def test_observed_pipe_replay_is_not_an_unbounded_actuator():
    path = Path(__file__).parents[2] / "configs" / "agc" / "agc2019_greenlight_observed_pipe_replay_v0.json"
    text = path.read_text()
    assert "min(railPipePeak" in text
    assert "min(growPipePeak" in text
    assert "not a counterfactual actuator model" in text
