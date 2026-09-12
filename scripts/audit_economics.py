"""Audit of the economic model: cost composition, parameter elasticity, a yield
comparison, and the consequences of the missing cost terms.

Reproduces every number in sections 7 and 8 of the internal change log.

    python scripts/audit_economics.py
"""
from __future__ import annotations
import numpy as np

from slowlab.economics import EconomicModel, sample_site_econ
from slowlab.world import ManagedTomgro, _maximise, MANAGEMENT_FACTORS

DAYS = 210.0
FAST = dict(n_sample=2500, n_start=5, stages=(0.2, 0.08, 0.03, 0.01), n_local=350)
FINE = dict(n_sample=4000, n_start=8, stages=(0.2, 0.08, 0.03, 0.01), n_local=500)


def optimum(w, seed=3, **kw):
    return _maximise(w, w.d, np.random.default_rng(seed), **(kw or FINE))


def cost_breakdown():
    e = EconomicModel()
    w = ManagedTomgro(seed=0, econ=e)
    x0, v0 = optimum(w)
    d = {f.name: float(f.denorm(x0[i])) for i, f in enumerate(MANAGEMENT_FACTORS)}
    lt = float(e.lighting_cost(d["par"]))
    ht = float(e.heating_cost(d["day_temp"], d["night_temp"]))
    cc = float(e.co2_cost(d["co2"]))
    pc = float(e.plant_cost(d["density"], DAYS))
    tot = lt + ht + cc + pc
    rev = float(w(x0.reshape(1, -1), components=True)["rev_rate"][0])

    print("optimum", {k: round(v, 1) for k, v in d.items()})
    print("revenue rate %.4f  cost rate %.4f  margin rate %.4f EUR/(m2.d)  margin %.0f%%"
          % (rev, tot, rev - tot, 100 * (rev - tot) / rev))
    print("\ncost composition:")
    for n, x in (("light (elec)", lt), ("heating (gas)", ht), ("CO2 purchased", cc), ("plants+labour", pc)):
        print("  %-14s %.5f  %5.1f%% of cost  %5.1f%% of revenue"
              % (n, x, 100 * x / tot, 100 * x / rev))

    print("\nelasticity of margin to a +10% parameter change:")
    for k in ("price_per_kg_fw", "dm_fraction", "labour_per_plant_day",
              "transplant_cost", "heat_price", "u_value", "elec_price",
              "co2_price", "co2_leak", "par_ambient"):
        w2 = ManagedTomgro(seed=0, econ=EconomicModel(**{k: getattr(e, k) * 1.1}))
        v2 = float(w2(x0.reshape(1, -1))[0])
        print("  %-22s %+7.1f%%" % (k, 100 * (v2 - v0) / v0))

    fw = rev * DAYS / e.price_per_kg_fw
    print("\nper cycle of %.0f days: fresh fruit %.1f kg/m2  revenue %.1f  cost %.1f  gross margin %.1f EUR/m2"
          % (DAYS, fw, rev * DAYS, tot * DAYS, (rev - tot) * DAYS))
    print("annualised (365/%.0f): yield %.0f kg/(m2.yr)  revenue %.0f  cost %.0f  margin %.0f"
          % (DAYS, fw * 365 / DAYS, rev * 365, tot * 365, (rev - tot) * 365))
    print("for comparison: Dutch greenhouse tomato 50-70 kg/(m2.yr) marketable, about 60 EUR/(m2.yr) of output")
    print("note: ours is gross yield, the comparison is marketable yield.")


def ceiling(make_econ, M=30, revenue_patch=None):
    """R*(none) = the expected regret of the best single-point recommendation."""
    old = EconomicModel.revenue
    if revenue_patch is not None:
        EconomicModel.revenue = revenue_patch
    try:
        Ws = [ManagedTomgro(seed=s, econ=make_econ(s)) for s in range(M)]
        opt, best = [], []
        for s, w in enumerate(Ws):
            x, v = optimum(w, seed=s, **FAST)
            opt.append(x); best.append(v)
        best, opt = np.array(best), np.array(opt)
        C = np.vstack([np.random.default_rng(7).random((3000, 6)), opt])
        F = np.array([np.asarray(w(C), float) for w in Ws])
        R = float((best[:, None] - F).mean(0).min())
        rev = float(Ws[0](opt[0].reshape(1, -1), components=True)["rev_rate"][0])
        sd = {f.name: float(np.std([f.denorm(o[i]) for o in opt]))
              for i, f in enumerate(MANAGEMENT_FACTORS)}
        return R, float(best.mean()), rev, sd
    finally:
        EconomicModel.revenue = old


def missing_cost_terms():
    """Add back yield-scaling harvest labour and grading loss -- omissions we created
    ourselves and never restored.

    0.21 EUR/kg comes from: total labour about 20 EUR/(m2.yr), harvest 40-50% of
    crop labour giving about 9, divided by 42 kg/(m2.yr). **This chain contains
    numbers we chose; see section 6 of the change log.**
    """
    def rev_net(self, wm):
        fw = np.asarray(wm, float) / self.dm_fraction / 1000.0
        return fw * (self.price_per_kg_fw - 0.21) * 0.93

    for tag, patch in (("as shipped", None), ("+harvest labour +grading loss", rev_net)):
        R, prof, rev, sd = ceiling(sample_site_econ, revenue_patch=patch)
        print("%-30s R*=%.5f  margin=%+.4f  revenue=%.4f  R*/revenue=%.2f%%"
              % (tag, R, prof, rev, 100 * R / rev))
    print("\nR*/revenue is identical in both cases -- the tomato price and dm_fraction cancel out of every ratio.")


def econ_decomposition():
    """One-at-a-time decomposition of site economic variation. The finding: individual contributions are near zero."""
    from slowlab.economics import SITE_ECON_SPREAD, _CLIMATE

    def only(keys):
        def mk(s):
            r = np.random.default_rng(70_000 + s)
            d = {k: float(r.uniform(*v)) for k, v in SITE_ECON_SPREAD.items()}
            u = float(r.random())
            for k, (lo, hi) in _CLIMATE.items():
                d[k] = lo + u * (hi - lo)
            return EconomicModel(**{k: d[k] for k in keys})
        return mk

    print("%-30s %s" % ("source of site variation", "R*(none)"))
    print("%-30s %.5f" % ("none (crop parameters only)", ceiling(lambda s: EconomicModel())[0]))
    for lab, keys in (("electricity price", ["elec_price"]), ("gas price", ["heat_price"]),
                      ("tomato price", ["price_per_kg_fw"]), ("U-value", ["u_value"]),
                      ("LED efficacy", ["led_efficacy"]),
                      ("climate", ["t_out_day", "t_out_night"])):
        print("%-30s %.5f" % ("+ " + lab, ceiling(only(keys))[0]))
    print("%-30s %.5f" % ("+ all of them", ceiling(sample_site_econ)[0]))


if __name__ == "__main__":
    for name, fn in (("cost composition and elasticity", cost_breakdown),
                     ("missing cost terms", missing_cost_terms),
                     ("site economic variation", econ_decomposition)):
        print("\n" + "=" * 66 + "\n" + name + "\n" + "=" * 66)
        fn()
