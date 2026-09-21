import math

import pytest

from slowlab.greenhouse_data import (
    clean_root_zone,
    excel_datetime,
    reconstruct_irrigation_events,
    water_balance_audit,
)


def test_excel_time_uses_declared_1900_epoch():
    assert excel_datetime(43815).isoformat().startswith("2019-12-16T00:00:00")


def test_irrigation_reconstruction_handles_midnight_without_negative_water():
    rows = [
        {"%time": "43815.9965278", "Cum_irr": "2.0"},
        {"%time": "43816.0000000", "Cum_irr": "0.2"},
        {"%time": "43816.0034722", "Cum_irr": "0.5"},
    ]
    events, audit = reconstruct_irrigation_events(rows, "Reference")
    assert [event.delivered_l_m2 for event in events] == pytest.approx([0.2, 0.3])
    assert audit["counter_resets"] == 1
    assert events[0].flag == "counter_reset"


def test_counter_reset_uses_new_counter_value_without_reusing_old_total():
    rows = [
        {"%time": "43815.1", "Cum_irr": "1.0"},
        {"%time": "43815.2", "Cum_irr": "0.2"},
        {"%time": "43815.3", "Cum_irr": "0.5"},
    ]
    events, audit = reconstruct_irrigation_events(rows, "Reference")
    assert [event.delivered_l_m2 for event in events] == pytest.approx([0.2, 0.3])
    assert audit["counter_resets"] == 1


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
    assert "WC_slab1:out_of_bounds" in record.flags


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
