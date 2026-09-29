from types import SimpleNamespace
import pytest
from slowlab.v22_greenlight_reuse import CropLifecycle, CROP_STATES, CROP_POOLS


def lifecycle():
    # Synthetic transition boundary, not a physical two-day simulation.
    x=CropLifecycle.__new__(CropLifecycle)
    x.crop_initial=dict(zip(CROP_STATES,(20000.,312.,4368.,1560.,0.,20.5,20.5)))
    state={**x.crop_initial,'tAir':24.,'tPipe':41.,'vpAir':1800.,'time_state':900.}
    x.engines={'active':SimpleNamespace(state=state,clock=900.),'empty':SimpleNamespace(state={},clock=0.)}
    x.mode='active';x.cleanup_seconds=172800.;x.ready_at=None;x.events=[]
    return x


def test_stop_preserves_facility_clock_and_logs_discard_without_revenue():
    x=lifecycle();before=dict(x.engine.state);x.stop()
    assert x.engine.clock==900.
    assert all(x.engine.state[k]==v for k,v in before.items() if k not in CROP_STATES)
    assert all(x.engine.state[k]==0 for k in CROP_POOLS)
    assert x.events[0]['removed_crop']['cFruit']==312.
    assert x.events[0]['standing_crop_revenue']==0.
    with pytest.raises(ValueError):x.stop()


def test_cleanup_and_replant_boundary():
    x=lifecycle();x.stop()
    x.engine.clock=x.ready_at-1
    with pytest.raises(ValueError):x.replant()
    x.engine.clock=x.ready_at
    x.engine.state.update(tAir=18.,tPipe=27.,vpAir=1300.,time_state=x.ready_at)
    before=dict(x.engine.state);now=x.engine.clock;x.replant()
    assert x.engine.clock==now
    assert all(x.engine.state[k]==v for k,v in before.items() if k not in CROP_STATES)
    assert x.engine.state['tCan']==x.engine.state['tCan24']==18.
    assert x.engine.state['tCanSum']==0.
    assert all(x.engine.state[k]==x.crop_initial[k] for k in CROP_POOLS)
    with pytest.raises(ValueError):x.replant()
