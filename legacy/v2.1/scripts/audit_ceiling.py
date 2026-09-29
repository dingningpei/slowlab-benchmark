"""Ceiling audit: the distribution of factor optima, a lambda sweep, the gate test,
and a solar-gain prototype.

Reproduces the numbers in sections 1, 5, 9 and 10 of the internal change log.

    python scripts/audit_ceiling.py
"""
from __future__ import annotations
import numpy as np

from slowlab.economics import EconomicModel, sample_site_econ
from slowlab.facility import Facility
from slowlab.tasks import TASKS, LABEL
from slowlab.world import ManagedTomgro, _maximise, MANAGEMENT_FACTORS

FAST = dict(n_sample=2500, n_start=5, stages=(0.2, 0.08, 0.03, 0.01), n_local=350)


def _opt(w, seed):
    return _maximise(w, w.d, np.random.default_rng(seed), **FAST)


def _profile(make_w, M=30, tag=""):
    Ws = [make_w(s) for s in range(M)]
    opt, best = [], []
    for s, w in enumerate(Ws):
        x, v = _opt(w, s)
        opt.append(x); best.append(v)
    best, opt = np.array(best), np.array(opt)
    C = np.vstack([np.random.default_rng(7).random((3000, 6)), opt])
    F = np.array([np.asarray(w(C), float) for w in Ws])
    R = float((best[:, None] - F).mean(0).min())
    print("%-22s R*(none)=%.5f  margin=%.4f  R*/margin=%.1f%%"
          % (tag, R, best.mean(), 100 * R / best.mean()))
    for i, f in enumerate(MANAGEMENT_FACTORS):
        r = np.array([f.denorm(o[i]) for o in opt])
        print("   %-11s %8.2f +/- %-6.2f [%.2f, %.2f]  on rail %.0f%%"
              % (f.name, r.mean(), r.std(), r.min(), r.max(),
                 100 * np.mean((opt[:, i] < 0.02) | (opt[:, i] > 0.98))))
    tc = np.array([w.params.values["TCRIT"] for w in Ws])
    dt = np.array([MANAGEMENT_FACTORS[0].denorm(o[0]) for o in opt])
    print("   corr(TCRIT, day_temp*) = %.2f" % np.corrcoef(tc, dt)[0, 1])
    return R


def lambda_scan(M=12):
    """How lambda determines where the optimum sits -- the evidence for removing it."""
    names = [f.name for f in MANAGEMENT_FACTORS]
    print("lam  " + " ".join("%10s" % n for n in names))
    for lam in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        rows = []
        for s in range(M):
            w = ManagedTomgro(seed=s, econ=EconomicModel(cost_weight=lam, strict=False))
            x, _ = _opt(w, s)
            rows.append([float(f.denorm(x[i])) for i, f in enumerate(MANAGEMENT_FACTORS)])
        A = np.array(rows)
        print("%4.2f " % lam + " ".join("%10.2f" % A[:, j].mean() for j in range(6)))
        print("     " + " ".join("%10s" % ("[%.1f,%.1f]" % (A[:, j].min(), A[:, j].max()))
                                 for j in range(6)))
    print("\ncommercial band: density 2.5-4 plants/m2, canopy LAI 3-4.5. Only lambda=1 lands inside.")


def gate():
    """The gate test: R*(none) against sigma/sqrt(B). It also exposes the gap between the declared noise_sd and the measured unit sd."""
    f = Facility(4, 3)
    print("plants_per_unit(density=3) = %d" % f.plants_per_unit(3.0))
    print("\n%-9s %12s %12s %7s %12s"
          % ("task", "declared_sd", "measured_sd", "overstated", "sig_tot/sqrt(B)"))
    # TASKS carries readable aliases besides T1-T4; iterate the canonical keys only.
    for k in ("T1", "T2", "T3", "T4"):
        t = TASKS[k]
        w = ManagedTomgro(seed=0, factors=t.factors, cycle_days=t.cycle_days)
        x0, _ = _maximise(w, len(t.factors), np.random.default_rng(3), **FAST)
        npl = f.plants_per_unit(3.0)
        rng = np.random.default_rng(0)
        vals = []
        for _ in range(60):
            pp = w.sample_plant_params(npl, rng, t.plant_cv)
            big = np.repeat(x0.reshape(1, -1), npl, axis=0)
            vals.append(float(np.asarray(w(big, plant_params=pp), float).mean()))
        sd = w.response_sd
        declared = t.plant_cv * sd
        tot = np.sqrt(declared ** 2 + (t.tau_chamber * sd) ** 2
                      + (t.tau_loop * sd) ** 2 + (t.tau_batch * sd) ** 2)
        B = t.n_rounds * t.units_per_round
        print("%-9s %12.5f %12.5f %6.1f× %12.5f"
              % (LABEL[k], declared, np.std(vals), declared / max(np.std(vals), 1e-9),
                 tot / np.sqrt(B)))
    print("\nthe declared value is plant_cv x response_sd, not what the simulation actually produces; the calibration table must be recomputed from measured values.")


def solar_gain_prototype():
    """Add solar gain to the heating model -- not yet merged; see section 5 of the
    change log.

    HL = A.U.dT is an *equipment sizing* formula that explicitly assumes no solar
    gain (Cornell / UGA extension material). The standard for operating energy is
    an energy balance with a solar term: Bot 1983 / Van Henten 1994 /
    de Zwart KASPRO 1996 / Vanthoor 2011, reviewed in Katzin et al. 2022,
    Agricultural Systems 198:103388.
    The PAR-to-broadband conversion of 1 MJ ~= 2.1 mol PAR is **still owed a citation**.
    """
    orig = EconomicModel.heating_cost

    def with_solar(self, t_day, t_night):
        dTd = np.clip(np.asarray(t_day, float) - self.t_out_day, 0, None)
        dTn = np.clip(np.asarray(t_night, float) - self.t_out_night, 0, None)
        gain = self.par_ambient / 2.1 / 3.6 * 1000.0          # Wh/(m²·d)
        wh = (np.clip(self.u_value * dTd * 12.0 - gain, 0, None)
              + self.u_value * dTn * 12.0)
        return wh / 1000.0 * self.heat_price * self.energy_shock

    e = EconomicModel()
    loss_d = e.u_value * (22.0 - e.t_out_day) * 12 / 1000.0
    loss_n = e.u_value * (18.0 - e.t_out_night) * 12 / 1000.0
    solar = e.par_ambient / 2.1 / 3.6
    print("daytime heat loss %.3f  night %.3f  transmitted solar %.3f kWh/(m2.d)" % (loss_d, loss_n, solar))
    print("solar gain is %.1fx the daytime heat loss and %.1fx the whole day's\n"
          % (solar / loss_d, solar / (loss_d + loss_n)))

    _profile(lambda s: ManagedTomgro(seed=s, econ=sample_site_econ(s)), tag="as shipped (no solar term)")
    EconomicModel.heating_cost = with_solar
    try:
        print()
        _profile(lambda s: ManagedTomgro(seed=s, econ=sample_site_econ(s)), tag="with solar gain")
    finally:
        EconomicModel.heating_cost = orig


if __name__ == "__main__":
    for name, fn in (("lambda sweep", lambda_scan),
                     ("gate test", gate),
                     ("solar-gain prototype (not merged)", solar_gain_prototype)):
        print("\n" + "=" * 66 + "\n" + name + "\n" + "=" * 66)
        fn()
