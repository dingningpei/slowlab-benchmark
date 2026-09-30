"""Cached compilation for the pinned numpy/LSODA GreenLight configuration.

Retains all equations. By default every auxiliary output is evaluated; with
``outputs`` (array-output mode) only the named outputs and their dependency
closure are, by the same statements in the same order, so the retained values
are identical. No timestep/cadence changes. Uses the already hash-verified
parser commands, never agent-provided code.
"""
import math
import re
import warnings

_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


def output_closure(model, outputs):
    """Names in ``model.solving_order`` needed to evaluate ``outputs``."""
    definitions = {k: model.variables_formatted[k] for k in model.solving_order}
    unknown = [k for k in outputs if k not in definitions and k != 'Time'
               and k not in model.states and k not in model.inputs]
    if unknown:
        raise ValueError('requested outputs are not model variables: ' + ', '.join(sorted(unknown)))
    needed, stack = set(), [k for k in outputs if k in definitions]
    while stack:
        key = stack.pop()
        if key in needed:
            continue
        needed.add(key)
        stack.extend(name for name in _NAME.findall(definitions[key])
                     if name in definitions and name not in needed)
    return [k for k in model.solving_order if k in needed]


class CachedGreenLightSolver:
    def __init__(self, model, *, array_output=False, outputs=None):
        import numpy as np
        self.model=model
        self.array_output=bool(array_output)
        if outputs is not None and not self.array_output:
            raise ValueError('selected outputs require array output')
        expected={'formatting_mode':'numpy','expand_variables':'False','solving_method':'solve_ivp_from_str',
                  'interpolation':'left','solver':'LSODA','t_eval':'None','clip_large_nums':'False','nans_to_zeros':'False'}
        if any(model.options[k]!=v for k,v in expected.items()):
            raise ValueError('cached solver configuration unsupported')
        self.signature=self._signature()
        # Same operations/order as upstream; omit only per-RHS progress print.
        source=('def rhs(t,y,d_matrix):\n'
                '    dy=np.zeros(y.shape)\n'
                '    row=np.searchsorted(d_matrix[:,0],t)-1\n'
                '    row=np.clip(row,0,d_matrix.shape[0]-1)\n'
                '    d=d_matrix[row,:]\n'
                f'    a=np.zeros({len(model.solving_order)})\n'+
                ''.join('    '+command+'\n' for command in model.commands)+'    return dy\n')
        scope={'np':np};exec(compile(source,'<cached-greenlight-rhs>','exec'),scope)
        self.rhs=scope['rhs']
        evaluated=list(model.solving_order) if outputs is None else output_closure(model,tuple(outputs))
        self.evaluated=tuple(evaluated)
        self.output_code=compile('\n'.join(k+' = '+model.variables_formatted[k] for k in evaluated),
                                 '<cached-greenlight-aux>','exec')

    def _signature(self):
        m=self.model
        return (tuple(m.states),tuple(m.inputs),tuple(m.commands),tuple(m.solving_order),
                tuple((k,m.variables_formatted[k]) for k in m.solving_order),
                tuple(sorted((k,v) for k,v in m.options.items() if k not in ('t_start','t_end'))))

    def solve(self, row=None):
        """Solve one segment. ``row`` (array-output mode) supplies the single
        current input row directly instead of through ``model.input_data``;
        the input matrix holds the same float64 values either way."""
        import numpy as np
        import pandas as pd
        from scipy.integrate import solve_ivp
        m=self.model
        if self._signature()!=self.signature:raise ValueError('compiled model changed; rebuild required')
        columns=['Time']+[k for k in m.inputs if k!='Time']
        if row is None:
            d=m.input_data[columns].to_numpy()
            current=lambda k: float(m.input_data.iloc[0][k])
        else:
            if not self.array_output:raise ValueError('direct input rows require array output')
            d=np.array([[float(row[k]) for k in columns]],dtype=np.float64)
            current=lambda k: float(row[k])
        if len(d)!=1:raise ValueError('cached adapter requires one present input row')
        t0,t1=float(m.options['t_start']),float(m.options['t_end'])
        if d[0,0]!=t0:raise ValueError('input timestamp must match current step')
        caught=[]
        if getattr(self.rhs,'emits_numpy_warnings',True) is False:
            # Compiled RHS: nothing inside can raise a numpy warning, so the
            # capture below would always record nothing.
            native=self.rhs
            def rhs(t,y):
                return native(t,y,d)
        else:
            def rhs(t,y):
                with np.errstate(all='warn'),warnings.catch_warnings(record=True) as logs:
                    warnings.simplefilter('always');out=self.rhs(t,y,d)
                for w in logs:
                    message=f'{w.category.__name__} encountered at time t={t}: {w.message}'
                    caught.append(message)
                    if m.options['warn_runtime'].lower()=='true':warnings.warn(message,w.category)
                return out
        try:first=float(m.options['first_step'])
        except ValueError:first=None
        sol=solve_ivp(rhs,[t0,t1],np.array([m.init[k] for k in m.states]),method='LSODA',
                      max_step=float(m.options['max_step']),first_step=first,
                      atol=float(m.options['atol']),rtol=float(m.options['rtol']))
        m.states_sol=sol
        if not sol.success:raise RuntimeError(sol.message)
        values={'Time':sol.t,'np':np}
        values.update({k:sol.y[i] for i,k in enumerate(m.states)})
        values.update({k:np.full(len(sol.t),current(k)) for k in columns[1:]})
        exec(self.output_code,values)
        order=['Time',*m.states,*columns[1:],*(k for k in self.evaluated if k in m.aux and k!='Time')]
        if self.array_output:
            # All native states and auxiliaries were still evaluated above; skip
            # only pandas packaging for the short-lived internal step result.
            arrays={}
            points=len(sol.t)
            for key in order:
                value=np.asarray(values[key],dtype=float)
                if value.ndim==0:
                    value=np.full(points,float(value))
                if value.shape!=(points,):
                    raise RuntimeError('invalid model output array: '+key)
                arrays[key]=value
            # One finiteness check over every output instead of one per array;
            # on failure name the first offending key, as before.
            if not np.isfinite(np.concatenate(tuple(arrays.values()))).all():
                bad=next(k for k,v in arrays.items() if not np.isfinite(v).all())
                raise RuntimeError('invalid model output array: '+bad)
            m.full_sol=arrays
        else:
            m.full_sol=pd.DataFrame({k:values[k] for k in order})
            if not np.isfinite(m.full_sol.to_numpy(dtype=float)).all():
                raise RuntimeError('nonfinite model output')
        m.log='\n'.join(dict.fromkeys(caught)) if m.options['log_runtime_warnings'].lower()=='true' else ''
