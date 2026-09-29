from datetime import date

from scripts.agc.audit_agc2019_crop_replay_state import latest_at_or_before


def test_crop_context_uses_latest_prior_measurement_without_future_leakage(tmp_path):
    path = tmp_path / "CropParameters.csv"
    path.write_text(
        "%Time,Stem_elong,Stem_thick,Cum_trusses,stem_dens ,plant_dens\n"
        "43887,33.7,10.8,12.4,4,1.4\n"
        "43894,31.3,10.4,13.7,4,1.4\n"
    )
    result = latest_at_or_before(path, date(2020, 3, 1))
    assert result["source_date"] == "2020-02-26"
    assert result["age_days_at_replay"] == 4
    assert result["observed"]["Cum_trusses"] == 12.4
    assert result["observed"]["stem_dens"] == 4
