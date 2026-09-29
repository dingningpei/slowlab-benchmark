import ast
import pytest
from slowlab.native_rhs import CEmitter

@pytest.mark.parametrize('source',['a[-1] = 0','a[10] = 0','a[0] = y[2]','y[0] = 0','a[0] = __import__("os").system("true")','a[0] = np.load("file")','a[0] = [1][0]','a[0] = np.exp(x=1)','a[0] = unknown','a[0] = float("nan")'])
def test_unsupported_syntax_and_out_of_bounds_fail_closed(source):
    with pytest.raises(ValueError):CEmitter({'a':10,'dy':2,'y':2,'d':3}).statement(ast.parse(source).body[0])

def test_integer_division_is_emitted_as_real_arithmetic():
    text=CEmitter({'a':1}).statement(ast.parse('a[0] = 1/2').body[0])
    assert '(1.0/2.0)' in text


def test_native_backend_requires_explicit_cached_solver():
    from slowlab.greenlight_reuse import ReusableGreenLight
    with pytest.raises(ValueError,match='requires cached'):
        ReusableGreenLight({},None,native_rhs=True)


def test_arithmetic_fault_preserves_finite_trial_diagnostic():
    import numpy as np
    from slowlab.native_rhs import NativeRHS
    rhs = NativeRHS.__new__(NativeRHS)
    rhs.states = 2
    rhs.inputs = 2
    rhs.function = lambda state, inputs, derivative: 8
    with pytest.raises(FloatingPointError, match='derivative_finite=True'):
        rhs(300.0, np.array([1.0, 2.0]), np.array([[0.0, 3.0]]))
    assert rhs.last_fault['flags'] == 8
    assert rhs.last_fault['state'] == [1.0, 2.0]
    assert rhs.last_fault['derivative_all_finite'] is True


def test_logistic_inverse_is_emitted_stably():
    text = CEmitter({'a':1,'y':1}).statement(
        ast.parse('a[0] = 1 / (1 + np.exp(y[0]))').body[0])
    assert 'inv_one_plus_exp(y[0])' in text


def test_native_logistic_extremes_are_finite_without_false_overflow():
    import numpy as np
    from types import SimpleNamespace
    from slowlab.native_rhs import NativeRHS
    model = SimpleNamespace(states=['x'], inputs=['Time'], solving_order=['a0'],
                            commands=['a[0] = 1 / (1 + np.exp(y[0]))',
                                      'dy[0] = a[0]'])
    rhs = NativeRHS(model)
    assert rhs(0., np.array([1000.]), np.array([[0.]]))[0] == 0.
    assert rhs(0., np.array([-1000.]), np.array([[0.]]))[0] == 1.
