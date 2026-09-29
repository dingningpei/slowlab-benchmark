"""Cached compilation for the pinned numpy/LSODA GreenLight configuration.

Retains all equations and all auxiliary outputs. No timestep/cadence changes.
Uses the already hash-verified parser commands, never agent-provided code.
"""
import math
import warnings


class CachedGreenLightSolver:
    def __init__(self, model, *, array_output=False):
        import numpy as np
        self.model=model
        self.array_output=bool(array_output)
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
        self.output_code=compile('\n'.join(k+' = '+model.variables_formatted[k] for k in model.solving_order),
                                 '<cached-greenlight-aux>','exec')

    def _signature(self):
        m=self.model
        return (tuple(m.states),tuple(m.inputs),tuple(m.commands),tuple(m.solving_order),
                tuple((k,m.variables_formatted[k]) for k in m.solving_order),
                tuple(sorted((k,v) for k,v in m.options.items() if k not in ('t_start','t_end'))))

    def solve(self):
        import numpy as np
        import pandas as pd
        from scipy.integrate import solve_ivp
        m=self.model
        if self._signature()!=self.signature:raise ValueError('compiled model changed; rebuild required')
        columns=['Time']+[k for k in m.inputs if k!='Time']
        d=m.input_data[columns].to_numpy()
        if len(d)!=1:raise ValueError('cached adapter requires one present input row')
        t0,t1=float(m.options['t_start']),float(m.options['t_end'])
        if d[0,0]!=t0:raise ValueError('input timestamp must match current step')
        caught=[]
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
        values.update({k:np.full(len(sol.t),float(m.input_data.iloc[0][k])) for k in columns[1:]})
        exec(self.output_code,values)
        order=['Time',*m.states,*columns[1:],*(k for k in m.solving_order if k in m.aux and k!='Time')]
        if self.array_output:
            # All native states and auxiliaries were still evaluated above; skip
            # only pandas packaging for the short-lived internal step result.
            arrays={}
            for key in order:
                value=np.asarray(values[key],dtype=float)
                if value.ndim==0:
                    value=np.full(len(sol.t),float(value))
                if value.shape!=(len(sol.t),) or not np.isfinite(value).all():
                    raise RuntimeError('invalid model output array: '+key)
                arrays[key]=value
            m.full_sol=arrays
        else:
            m.full_sol=pd.DataFrame({k:values[k] for k in order})
            if not np.isfinite(m.full_sol.to_numpy(dtype=float)).all():
                raise RuntimeError('nonfinite model output')
        m.log='\n'.join(dict.fromkeys(caught)) if m.options['log_runtime_warnings'].lower()=='true' else ''
