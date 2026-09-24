import csv
from scripts.run_agc2019_calibration_sequences import aggregate, sequence_metrics


def test_sequence_metrics_does_not_reset_at_midnight(tmp_path):
    sim=tmp_path/"sim.csv";obs=tmp_path/"obs.csv"
    with sim.open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["Time","tAir","rhIn","co2InPpm"]);w.writerow(["d","d","d","d"]);w.writerow(["s","C","%","ppm"])
        for h in range(48):w.writerow([h*3600,20+h/100,80,500])
    with obs.open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["timestamp","air_temperature_c","relative_humidity_pct","co2_ppm"])
        from datetime import datetime,timedelta
        start=datetime(2020,1,1)
        for h in range(48):w.writerow([(start+timedelta(hours=h)).isoformat(),20+h/100,80,500])
    from datetime import date
    result=sequence_metrics(sim,obs,date(2020,1,1),2)
    assert result["samples"]==48
    assert result["metrics"]["tAir"]["rmse"]==0


def test_aggregate_weights_by_sample_count():
    def record(name, n, rmse):
        metric={"mae":rmse,"rmse":rmse,"bias":rmse,"hourly_change_rmse":rmse}
        return {"compartment":name,"samples":n,"solver_complete":True,
                "physical_violations":{"tAir":0,"rhIn":0,"co2InPpm":0},
                "metrics":{key:dict(metric) for key in ("tAir","rhIn","co2InPpm")}}
    result=aggregate([record("A",2,1),record("A",8,3)])
    assert abs(result["pooled"]["metrics"]["tAir"]["rmse"]-(7.4**0.5)) < 1e-12
    assert result["A"]["samples"] == 10
