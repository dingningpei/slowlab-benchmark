import math
from types import SimpleNamespace

import pytest

from slowlab.v22.online_observations import OnlineObservations, PackedOnlineObservations
from slowlab.v22.greenlight_adapter import greenlight_raw_endpoint


def store():
    return OnlineObservations({'tAir': 'degC'})


def test_off_grid_query_never_interpolates_or_relabels():
    log = store()
    log.record('a', 'tAir', measurement_time=0, available_at=0, value=10)
    log.advance_to(300)
    log.record('a', 'tAir', measurement_time=300, available_at=300, value=20)
    assert log.latest('a', 'tAir', as_of=150)['value'] == 10
    assert log.latest('a', 'tAir', as_of=150)['measurement_time'] == 0
    assert log.latest('a', 'tAir', as_of=300)['value'] == 20


def test_delayed_out_of_order_delivery_and_no_first_future_clamp():
    log = store()
    log.record('a', 'tAir', measurement_time=0, available_at=400, value=10)
    assert log.latest('a', 'tAir') is None
    log.advance_to(300)
    log.record('a', 'tAir', measurement_time=300, available_at=300, value=20)
    assert log.latest('a', 'tAir')['value'] == 20
    log.advance_to(400)
    assert log.latest('a', 'tAir')['value'] == 20
    assert [r['measurement_time'] for r in log.history('a', 'tAir')] == [0, 300]
    assert len(log.history('a', 'tAir', as_of=300)) == 1


def test_future_suffix_and_repeated_reads_do_not_change_prefix():
    outputs = []
    for future in [-999, 999]:
        log = store()
        log.record('a', 'tAir', measurement_time=0, available_at=0, value=10)
        original = log.history('a', 'tAir')
        log.advance_to(300)
        log.record('a', 'tAir', measurement_time=300, available_at=300, value=future)
        outputs.append(log.history('a', 'tAir', as_of=150))
        assert outputs[-1] == original
        payload = log.latest('a', 'tAir', as_of=150)
        payload['value'] = 123456
        assert log.latest('a', 'tAir', as_of=150)['value'] == 10
    assert outputs[0] == outputs[1]


def test_immutable_identities_and_backdated_delivery_rejected():
    log = store()
    log.record('a', 'tAir', measurement_time=0, available_at=0, value=10)
    with pytest.raises(ValueError, match='immutable'):
        log.record('a', 'tAir', measurement_time=0, available_at=0, value=11)
    log.advance_to(300)
    with pytest.raises(ValueError, match='backdated'):
        log.record('b', 'tAir', measurement_time=0, available_at=0, value=10)
    with pytest.raises(ValueError, match='already occurred'):
        log.record('a', 'tAir', measurement_time=600, available_at=600, value=10)
    with pytest.raises(ValueError, match='backwards'):
        log.advance_to(0)
    with pytest.raises(ValueError):
        log.latest('a', 'tAir', as_of=301)
    with pytest.raises(ValueError):
        log.latest('a', 'private_crop_truth')
    assert log.latest('other', 'tAir') is None


def test_late_out_of_order_identity_remains_immutable():
    log = store()
    log.advance_to(300)
    log.record('a', 'tAir', measurement_time=300, available_at=300, value=20)
    log.record('a', 'tAir', measurement_time=150, available_at=300, value=15)
    with pytest.raises(ValueError, match='immutable'):
        log.record('a', 'tAir', measurement_time=150, available_at=300, value=99)
    assert [r['measurement_time'] for r in log.history('a', 'tAir')] == [150, 300]


def test_packed_store_preserves_delayed_and_historical_queries():
    stores = [cls({'tAir': 'degC'}) for cls in
              (OnlineObservations, PackedOnlineObservations)]
    for log in stores:
        log.record('a', 'tAir', measurement_time=0, available_at=600, value=10)
        log.advance_to(300)
        log.record('a', 'tAir', measurement_time=300, available_at=300, value=20)
        log.advance_to(600)
        log.record('a', 'tAir', measurement_time=150, available_at=600, value=15)
        log.record('a', 'tAir', measurement_time=600, available_at=600, value=30)
        with pytest.raises(ValueError, match='immutable'):
            log.record('a', 'tAir', measurement_time=150, available_at=600, value=99)
    for cutoff in (0, 150, 300, 450, 600):
        assert stores[0].latest('a', 'tAir', as_of=cutoff) == stores[1].latest('a', 'tAir', as_of=cutoff)
        assert stores[0].history('a', 'tAir', as_of=cutoff) == stores[1].history('a', 'tAir', as_of=cutoff)
    assert stores[0].history('a', 'tAir', start=150, end=300) == stores[1].history('a', 'tAir', start=150, end=300)
    copied = stores[1].latest('a', 'tAir')
    copied['value'] = -999
    assert stores[1].latest('a', 'tAir')['value'] == 30


@pytest.mark.parametrize('bad', [math.nan, math.inf, -math.inf, True, '300'])
def test_nonfinite_or_ambiguous_times_rejected(bad):
    with pytest.raises(ValueError):
        store().advance_to(bad)


def solution(end=300, value=20, success=True):
    return SimpleNamespace(success=success, t=[0, end], y=[[10, value]])


def test_raw_endpoint_bridge_and_delivery():
    log = store()
    log.advance_to(300)
    sample = greenlight_raw_endpoint(solution(), ['tAir'], ['tAir'], expected_time=300)
    log.record('a', 'tAir', measurement_time=300, available_at=310, value=sample['tAir'])
    assert log.latest('a', 'tAir') is None
    log.advance_to(310)
    assert log.latest('a', 'tAir')['value'] == 20
    assert log.latest('a', 'tAir')['measurement_time'] == 300


@pytest.mark.parametrize('bad', [solution(end=600), solution(end=299),
                                solution(success=False), solution(value=math.nan)])
def test_raw_bridge_rejects_future_partial_or_invalid_solution(bad):
    with pytest.raises(ValueError):
        greenlight_raw_endpoint(bad, ['tAir'], ['tAir'], expected_time=300)


def test_raw_bridge_rejects_csv_and_invalid_shape():
    with pytest.raises(ValueError):
        greenlight_raw_endpoint('offline.csv', ['tAir'], ['tAir'], expected_time=300)
    with pytest.raises(ValueError):
        greenlight_raw_endpoint(solution(), ['tAir', 'tAir'], ['tAir'], expected_time=300)
    with pytest.raises(ValueError):
        greenlight_raw_endpoint(solution(), ['tAir'], ['co2Air'], expected_time=300)
