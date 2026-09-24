import math
import json
from pathlib import Path

import pytest

from slowlab.greenhouse_data import (
    audit_production_dates,
    clean_climate_observations,
    clean_root_zone,
    excel_datetime,
    reconstruct_irrigation_events,
    reconstruct_pump_minutes,
    water_balance_audit,
)


def test_excel_time_uses_declared_1900_epoch():
    assert excel_datetime(43815).isoformat().startswith("2019-12-16T00:00:00")
    assert excel_datetime(43815).tzinfo is None


def test_irrigation_reconstruction_handles_scheduled_reset_without_negative_water():
    rows = [
        {"%time": "43816.0208333", "Cum_irr": "2.0"},
        {"%time": "43816.0243056", "Cum_irr": "0.2"},
        {"%time": "43816.0277778", "Cum_irr": "0.5"},
    ]
    events, audit = reconstruct_irrigation_events(rows, "Reference")
    assert [event.delivered_l_m2 for event in events] == pytest.approx([0.2, 0.3])
    assert audit["counter_resets"] == 1
    assert events[0].flag == "counter_reset"


def test_counter_reset_uses_new_counter_value_without_reusing_old_total():
    rows = [
        {"%time": "43815.0208333", "Cum_irr": "1.0"},
        {"%time": "43815.0243056", "Cum_irr": "0.2"},
        {"%time": "43815.0277778", "Cum_irr": "0.5"},
    ]
    events, audit = reconstruct_irrigation_events(rows, "Reference")
    assert [event.delivered_l_m2 for event in events] == pytest.approx([0.2, 0.3])
    assert audit["counter_resets"] == 1


def test_unscheduled_counter_correction_is_not_counted_as_water():
    rows = [
        {"%time": "43815.10", "Cum_irr": "0.6"},
        {"%time": "43815.11", "Cum_irr": "0.3"},
        {"%time": "43815.12", "Cum_irr": "0.6"},
        {"%time": "43815.13", "Cum_irr": "0.9"},
    ]
    events, audit = reconstruct_irrigation_events(rows, "Reference")
    assert [event.delivered_l_m2 for event in events] == pytest.approx([0.3])
    assert audit["counter_resets"] == 0
    assert audit["counter_corrections"] == 1


def test_agc_initial_partial_counter_cycle_can_be_discarded():
    rows = [
        {"%time": "43815.0", "Cum_irr": "31.6"},
        {"%time": "43815.01", "Cum_irr": "31.9"},
        {"%time": "43815.0243056", "Cum_irr": "0.1"},
        {"%time": "43815.03", "Cum_irr": "0.4"},
    ]
    events, audit = reconstruct_irrigation_events(
        rows, "AICU", discard_initial_partial_cycle=True
    )
    assert [event.delivered_l_m2 for event in events] == pytest.approx([0.1, 0.3])
    assert audit["initial_partial_cycle_rows"] == 2


def test_raw_pump_minutes_are_reconstructed_without_litres_conversion():
    rows = [
        {"%time": "43815.0", "water_sup": "37", "Cum_irr": "3.7"},
        {"%time": "43815.01", "water_sup": "39", "Cum_irr": "3.9"},
        {"%time": "43815.0243056", "water_sup": "1", "Cum_irr": "0.1"},
        {"%time": "43815.03", "water_sup": "4", "Cum_irr": "0.4"},
        {"%time": "43815.04", "water_sup": "3", "Cum_irr": "0.3"},
        {"%time": "43815.05", "water_sup": "5", "Cum_irr": "0.5"},
    ]
    events, audit = reconstruct_pump_minutes(
        rows, "Reference", discard_initial_partial_cycle=True
    )
    assert [event.pump_minutes for event in events] == pytest.approx([1, 3, 1])
    assert audit["counter_resets"] == 1
    assert audit["counter_corrections"] == 1
    assert events[0].flag == "counter_reset"


def test_short_missing_counter_block_does_not_erase_accumulated_delivery():
    rows = [
        {"%time": "43815.1", "Cum_irr": "1.0"},
        {"%time": "43815.2", "Cum_irr": "NaN"},
        {"%time": "43815.3", "Cum_irr": "1.4"},
    ]
    events, audit = reconstruct_irrigation_events(rows, "Reference")
    assert [event.delivered_l_m2 for event in events] == pytest.approx([0.4])
    assert audit["missing_counter"] == 1
    assert audit["resumed_after_missing"] == 1


def test_root_zone_rejects_zero_sensor_sentinels_but_keeps_missing_distinct():
    rows = [{
        "%time": "43815.0",
        "WC_slab1": "0",
        "WC_slab2": "75",
        "EC_slab1": "0",
        "EC_slab2": "5.1",
        "t_slab1": "NaN",
        "t_slab2": "21",
    }]
    records, audit = clean_root_zone(rows, "Digilog")
    record = records[0]
    assert record.wc_slab1_pct is None
    assert record.ec_slab1_ds_m is None
    assert record.wc_slab2_pct == 75
    assert record.t_slab1_c is None
    assert audit["rejected_values"] == 2
    assert audit["rejected_by_field"] == {"WC_slab1": 1, "EC_slab1": 1}
    assert "WC_slab1:out_of_bounds" in record.flags


def test_climate_observations_flag_negative_reference_sensor_values():
    rows = [{
        "%time": "43861.48958", "Tair": "-1", "Rhair": "-22.2",
        "CO2air": "-499",
    }]
    records, audit = clean_climate_observations(rows, "Reference")
    assert len(records) == 1
    assert records[0].air_temperature_c is None
    assert records[0].relative_humidity_pct is None
    assert records[0].co2_ppm is None
    assert audit["rejected_by_field"] == {"Tair": 1, "Rhair": 1, "CO2air": 1}


def test_reference_production_year_repair_retains_source_serial():
    rows = [{"%time": "43510"}, {"%time": "43880"}]
    records, audit = audit_production_dates(rows, "Reference")
    assert records[0].source_excel_time == 43510
    assert records[0].corrected_excel_time == 43875
    assert records[0].correction_flag == "source_year_offset_plus_365_days"
    assert records[1].corrected_excel_time == 43880
    assert audit["year_repair"] == 1


def test_daily_water_audit_compares_reconstructed_and_reported_volume():
    rows = [
        {"%time": "43815.0", "Cum_irr": "0.2"},
        {"%time": "43815.1", "Cum_irr": "0.6"},
    ]
    events, _ = reconstruct_irrigation_events(rows, "Reference")
    audit = water_balance_audit(events, {
        43815: {"irrigation_l_m2": 0.7, "drain_l_m2": 0.1}
    })
    assert len(audit) == 1
    assert audit[0]["reconstructed_irrigation_l_m2"] == pytest.approx(0.4)
    assert audit[0]["difference_l_m2"] == pytest.approx(-0.3)
    assert audit[0]["reported_drain_l_m2"] == pytest.approx(0.1)


def test_frozen_root_zone_cleaning_config_matches_reset_rule():
    config = json.loads(
        (Path(__file__).parents[1] / "configs" / "agc2019_root_zone_cleaning_v2.json").read_text()
    )
    assert config["version"] == 2
    assert config["irrigation_counter"]["reset_tolerance_minutes"] == 3
    assert config["digilog_zero_audit"]["zero_values"] == 98
