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
