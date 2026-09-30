"""Experimental native RHS for hash-verified GreenLight parser commands.

Strict AST allowlist; no arbitrary code or shell interpolation. Compiles locally
with a pinned local clang or GCC path, no fast-math/FMA contraction. Reference solver remains the default.
"""
import ast
import ctypes
import hashlib
import math
import platform
import shutil
from pathlib import Path
import subprocess
import tempfile


class CEmitter:
    def __init__(self, sizes):
        self.sizes=sizes

    def expr(self,n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float) and math.isfinite(n.value):
            return repr(float(n.value))
        if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id in self.sizes:
            if not isinstance(n.slice,ast.Constant) or type(n.slice.value) is not int or not 0<=n.slice.value<self.sizes[n.value.id]:
                raise ValueError('unsafe array index')
            return f'{n.value.id}[{n.slice.value}]'
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return '(-'+self.expr(n.operand)+')'
        if isinstance(n,ast.BinOp):
            # Algebraically stable 1/(1+exp(x)). Native libm raises FE_OVERFLOW
            # for large positive x even though the complete expression is zero.
            # The pinned reference RHS has the same benign intermediate warning.
            if (isinstance(n.op,ast.Div) and isinstance(n.left,ast.Constant)
                    and type(n.left.value) in (int,float) and n.left.value==1
                    and isinstance(n.right,ast.BinOp) and isinstance(n.right.op,ast.Add)):
                for one, exponential in ((n.right.left,n.right.right),
                                         (n.right.right,n.right.left)):
                    if (isinstance(one,ast.Constant) and type(one.value) in (int,float)
                            and one.value==1 and isinstance(exponential,ast.Call)
                            and not exponential.keywords and len(exponential.args)==1
                            and isinstance(exponential.func,ast.Attribute)
                            and isinstance(exponential.func.value,ast.Name)
                            and exponential.func.value.id=='np'
                            and exponential.func.attr=='exp'):
                        return 'inv_one_plus_exp('+self.expr(exponential.args[0])+')'
            left,right=self.expr(n.left),self.expr(n.right)
            if isinstance(n.op,ast.Pow):return f'pow({left},{right})'
            op={ast.Add:'+',ast.Sub:'-',ast.Mult:'*',ast.Div:'/'} .get(type(n.op))
            if op:return f'({left}{op}{right})'
        if isinstance(n,ast.Compare) and len(n.ops)==1:
            op={ast.Gt:'>',ast.GtE:'>=',ast.Lt:'<',ast.LtE:'<=',ast.NotEq:'!=',ast.Eq:'=='}.get(type(n.ops[0]))
            if op:return f'({self.expr(n.left)}{op}{self.expr(n.comparators[0])})'
        if isinstance(n,ast.Call) and not n.keywords and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='np':
            functions={'abs':('fabs',1),'exp':('exp',1),'log':('log',1),'sqrt':('sqrt',1),'cos':('cos',1),'mod':('numpy_mod',2),'logical_and':('numpy_and',2),'logical_or':('numpy_or',2)}
            if n.func.attr in functions:
                name,count=functions[n.func.attr]
                if len(n.args)==count:return name+'('+','.join(self.expr(a) for a in n.args)+')'
        raise ValueError('unsupported native expression: '+ast.dump(n))

    def statement(self,node):
        if not isinstance(node,ast.Assign) or len(node.targets)!=1:raise ValueError('expected one assignment')
        target=node.targets[0]
        if not isinstance(target,ast.Subscript) or not isinstance(target.value,ast.Name) or target.value.id not in ('a','dy'):
            raise ValueError('only auxiliary/derivative assignments permitted')
        return self.expr(target)+' = '+self.expr(node.value)+';'


class NativeRHS:
    def __init__(self,model):
        import numpy as np
        self.states=len(model.states);self.inputs=1+sum(k!='Time' for k in model.inputs)
        emit=CEmitter({'y':self.states,'dy':self.states,'d':self.inputs,'a':len(model.solving_order)})
        statements='\n'.join(emit.statement(n) for n in ast.parse('\n'.join(model.commands)).body)
        source='''#include <math.h>
#include <fenv.h>
#pragma STDC FENV_ACCESS ON
static double inv_one_plus_exp(double x) {
    if (x >= 0) { double z = exp(-x); return z / (1 + z); }
    double z = exp(x); return 1 / (1 + z);
}
static double numpy_mod(double x,double y) { double r=fmod(x,y); if(r!=0 && ((r<0)!=(y<0))) r+=y; return r==0 ? copysign(0.0,y) : r; }
static double numpy_and(double x,double y) {return (x!=0) && (y!=0);}
static double numpy_or(double x,double y) {return (x!=0) || (y!=0);}
int rhs(const double *y,const double *d,double *dy) {
feclearexcept(FE_ALL_EXCEPT);
'''+f'double a[{len(model.solving_order)}]={{0}};\n'+statements+'''
return fetestexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
}
/* Checked entry point: the same rhs() above, bracketed by finiteness checks.
   isfinite() only classifies values and raises no floating-point exception. */
int rhs_checked(const double *y,const double *d,double *dy,int ny,int nd) {
for (int i=0;i<ny;i++) if (!isfinite(y[i])) return -1;
for (int i=0;i<nd;i++) if (!isfinite(d[i])) return -1;
int flags=rhs(y,d,dy);
for (int i=0;i<ny;i++) if (!isfinite(dy[i])) return flags | (1<<30);
return flags;
}
'''
        self.source_sha256=hashlib.sha256(source.encode()).hexdigest()
        self._temporary=tempfile.TemporaryDirectory(prefix='slowlab-native-rhs-')
        directory=Path(self._temporary.name);src=directory/'rhs.c'
        system=platform.system()
        if system not in ('Darwin','Linux'):
            raise RuntimeError('native RHS supports only audited Darwin/Linux compilers')
        lib=directory/('rhs.dylib' if system=='Darwin' else 'rhs.so')
        src.write_text(source)
        compiler=shutil.which('clang') or shutil.which('gcc')
        if compiler is None:
            raise RuntimeError('native RHS requires clang or GCC')
        self.compiler_path=str(Path(compiler).resolve())
        self.compiler_flags=['-O2','-fno-fast-math','-ffp-contract=off','-shared','-fPIC']
        link_flags=['-lm'] if system=='Linux' else []
        subprocess.run([self.compiler_path,*self.compiler_flags,str(src),'-o',str(lib),*link_flags],
                       check=True,capture_output=True,timeout=30)
        self.compiler_version=subprocess.run([self.compiler_path,'--version'],check=True,capture_output=True,text=True).stdout.splitlines()[0]
        self.library=ctypes.CDLL(str(lib));self.function=self.library.rhs_checked
        self.function.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2;self.function.restype=ctypes.c_int
        self._allocate()

    def _allocate(self):
        """Persistent buffers with prebuilt pointers: a call copies y and the
        current input row into them instead of converting three numpy
        arrays through ctypes each time. The arithmetic in rhs() is unchanged."""
        import numpy as np
        self._y=np.zeros(self.states,dtype=np.float64);self._d=np.zeros(self.inputs,dtype=np.float64)
        self._out=np.zeros(self.states,dtype=np.float64)
        self._pointers=tuple(ctypes.c_void_p(a.ctypes.data) for a in (self._y,self._d,self._out))
        self._sizes=(ctypes.c_int(self.states),ctypes.c_int(self.inputs))

    # C code raises no numpy floating-point warnings, so the cached solver may
    # call this without its per-call numpy warning capture.
    emits_numpy_warnings = False

    def __call__(self,t,y,d_matrix):
        row=d_matrix[0]
        if y.shape!=(self.states,) or row.shape!=(self.inputs,):raise ValueError('native RHS shape mismatch')
        # Copy the input row on every call: callers may mutate d_matrix in place.
        self._d[:]=row
        self._y[:]=y
        self._out[:]=0.0
        code=self.function(*self._pointers,*self._sizes)
        if code==-1:raise ValueError('nonfinite native input')
        if code:
            flags=code & ~(1<<30);output_finite=not code & (1<<30)
            # Preserve the exact failed LSODA trial evaluation for diagnosis.
            # This is executor-private and never included in agent payloads.
            self.last_fault = {'time': float(t), 'flags': int(flags),
                               'derivative_all_finite': output_finite,
                               'state': self._y.tolist(), 'inputs': self._d.tolist(),
                               'derivative': [float(v) if v==v and abs(v)!=float('inf') else None for v in self._out.tolist()]}
            raise FloatingPointError(f'native RHS arithmetic fault: {flags}; derivative_finite={output_finite}')
        return self._out.copy()
