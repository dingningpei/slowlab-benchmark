from types import SimpleNamespace
import pytest
from slowlab.greenlight_reuse import require_dynamic_inputs


def test_json_input_auxiliaries_cannot_silently_run_as_dynamic_controls():
    model=SimpleNamespace(inputs={},aux={'cmdHeat':'0','cmdLamp':'0'})
    with pytest.raises(RuntimeError,match='not registered'):
        require_dynamic_inputs(model,['cmdHeat','cmdLamp'])


def test_partial_registration_is_rejected():
    with pytest.raises(RuntimeError,match='cmdLamp'):
        require_dynamic_inputs(SimpleNamespace(inputs={'cmdHeat':'cmdHeat'}),['cmdHeat','cmdLamp'])


def test_required_registered_channels_pass_guard():
    require_dynamic_inputs(SimpleNamespace(inputs={'cmdHeat':'cmdHeat','cmdLamp':'cmdLamp'}),['cmdHeat','cmdLamp'])


@pytest.mark.parametrize('leaf', [0., -1., float('nan'), float('inf')])
def test_unsupported_empty_or_invalid_canopy_fails_before_solving(leaf):
    from slowlab.greenlight_reuse import require_live_canopy
    with pytest.raises(RuntimeError, match='crop-absent mode'):
        require_live_canopy({'cLeaf': leaf})


def test_live_canopy_remains_supported():
    from slowlab.greenlight_reuse import require_live_canopy
    require_live_canopy({'cLeaf': 4368.})
