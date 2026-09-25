import json
from pathlib import Path
from scripts.build_agc2019_observed_pipe_replay_v1 import build


def test_feedforward_compensates_heat_loss_and_remains_capacity_bounded(tmp_path):
    data=build(Path("configs/agc2019_greenlight_observed_pipe_replay_v0.json"),tmp_path/"v1.json")
    tracking=data["Internal pipe state tracking"]
    rail=tracking["hBoilPipe"]["definition"]
    grow=tracking["hBoilGroPipe"]["definition"]
    assert "hObservedRailPipeDemand +" in rail and "min(railPipePeak" in rail
    assert "hObservedGrowPipeDemand +" in grow and "min(growPipePeak" in grow
    assert "pipeLowActive *" in rail and "pipeGrowActive *" in grow
    assert json.loads((tmp_path/"v1.json").read_text()) == data
