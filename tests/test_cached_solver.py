from types import SimpleNamespace
import pytest
from slowlab.cached_solver import CachedGreenLightSolver


def model():
    return SimpleNamespace(options={'formatting_mode':'numpy','expand_variables':'False','solving_method':'solve_ivp_from_str','interpolation':'left','solver':'LSODA','t_eval':'None','clip_large_nums':'False','nans_to_zeros':'False'},states={'x':'x'},inputs={'u':'u'},commands=['dy[0] = d[1] - y[0]'],solving_order=['flow'],variables_formatted={'flow':'x'})

@pytest.mark.parametrize('key,value',[('interpolation','linear'),('nans_to_zeros','True'),('clip_large_nums','True'),('formatting_mode','math'),('t_eval','yes'),('solver','BDF')])
def test_unsupported_modes_fail_closed(key,value):
    m=model();m.options[key]=value
    with pytest.raises(ValueError,match='unsupported'):CachedGreenLightSolver(m)


def test_compiled_signature_detects_changed_equations_and_options():
    m=model();cache=CachedGreenLightSolver(m)
    m.options['t_start']='300';m.options['t_end']='600'
    assert cache._signature()==cache.signature
    m.variables_formatted['flow']='2*x'
    assert cache._signature()!=cache.signature
    m.variables_formatted['flow']='x';m.options['solver']='BDF'
    assert cache._signature()!=cache.signature


def fake_model():
    from types import SimpleNamespace
    order = ['a', 'b', 'c', 'd', 'e']
    formatted = {'a': 'x * 2', 'b': 'a + 1', 'c': 'np.exp(b)', 'd': 'x - 1', 'e': 'c + d'}
    return SimpleNamespace(solving_order=order, variables_formatted=formatted,
                           states=['x'], inputs=['Time', 'u'], aux=set(order))


def test_output_closure_follows_indirect_dependencies_in_solving_order():
    from slowlab.cached_solver import output_closure
    model = fake_model()
    assert output_closure(model, ('c',)) == ['a', 'b', 'c']
    assert output_closure(model, ('e',)) == ['a', 'b', 'c', 'd', 'e']
    assert output_closure(model, ('d', 'Time', 'x', 'u')) == ['d']


def test_output_closure_rejects_unknown_names():
    import pytest
    from slowlab.cached_solver import output_closure
    with pytest.raises(ValueError, match='not model variables'):
        output_closure(fake_model(), ('nope',))


def test_pruned_statements_give_identical_values_to_the_full_set():
    import numpy as np
    from slowlab.cached_solver import output_closure
    model = fake_model()
    x = np.linspace(-1.0, 1.0, 7)
    full = {'x': x, 'np': np}
    exec('\n'.join(k + ' = ' + model.variables_formatted[k] for k in model.solving_order), full)
    kept = output_closure(model, ('c',))
    pruned = {'x': x, 'np': np}
    exec('\n'.join(k + ' = ' + model.variables_formatted[k] for k in kept), pruned)
    assert np.array_equal(pruned['c'], full['c']) and 'e' not in pruned


def test_executor_requests_only_its_consumed_outputs_unless_asked_for_all():
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from slowlab.campaign_executor import EXECUTOR_OUTPUTS, CampaignExecutor
    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root / 'configs/task_contract_v4.json').read_text())
    policy = json.loads((root / 'configs/campaign_example_v0.json').read_text())['policy_a']
    seen = []

    class Capture:
        def __init__(self, contract, source, start, **kwargs):
            seen.append(kwargs['outputs'])
            self.clock, self.mode, self.ready_at = 0.0, 'empty', 0.0
            self.model = SimpleNamespace(full_sol={}, input_data=[{}])

        @property
        def engine(self):
            return self

    for everything, expected in ((False, EXECUTOR_OUTPUTS), (True, None)):
        seen.clear()
        CampaignExecutor(contract, Path('/unused'), None, feedback_mode='full', fallback_policy=policy,
                         lifecycle_factory=Capture, sample_endpoint=lambda *a: None,
                         all_model_outputs=everything)
        assert seen == [expected] * 4
    assert set(EXECUTOR_OUTPUTS) == {'hBoilPipe', 'qLampIn', 'mcExtAir', 'mcFruitHar', 'mvCanAir', 'rhIn', 'co2InPpm'}


def test_solver_end_time_snap_is_bounded():
    import numpy as np
    from types import SimpleNamespace
    from slowlab.cached_solver import snap_end_time
    t1 = 13787400.0
    for offset, expect in ((2.744e-5, True), (-5e-4, True), (0.0, False), (5e-3, False)):
        sol = SimpleNamespace(success=True, t=np.array([t1 - 300.0, t1 + offset]))
        assert snap_end_time(sol, t1) is expect
        assert bool(sol.t[-1] == t1) is (expect or offset == 0.0)
    failed = SimpleNamespace(success=False, t=np.array([0.0, 300.00001]))
    assert snap_end_time(failed, 300.0) is False and bool(failed.t[-1] != 300.0)
