"""Greenhouse gross-margin model: marketable yield minus the cost of growing it.

Why this exists
---------------
Under a pure yield objective, TOMGRO is *monotone* in CO2, supplemental light
and night temperature -- more is always better, the optimum sits on the upper
rail of those factors, and the search problem is non-trivial in temperature
alone.

Real greenhouse optimisation is never pure yield. The literature phrases it as
"maximize yield **under acceptable energy consumption**". Once the cost term is
added, every resource factor acquires an interior optimum: a little more pays,
a lot more does not.

Why lambda (cost_weight) was removed
------------------------------------
An earlier version multiplied cost by lambda and called it a "task design
parameter", defaulting to 0.5 (0.25 during T4 training). That was the single
worst piece of data manipulation in this project, because it **sets where the
optimum is**. Measured over 12 sites, joint optimum in six factors:

    lambda   day_temp  night_temp   co2    par    density  lai_max
    0.25      22.9     22.0 RAIL    1164   32 RAIL  8.0 RAIL  4.95
    0.50      22.4     21.1          760   19.9      6.36     4.90
    1.00      22.3     17.8          452   18 RAIL   3.26     4.18

Commercial greenhouse tomato density is 2.5-4 plants/m^2 and canopy LAI 3-4.5.
lambda=1 lands inside that; lambda=0.5 is twice it; at lambda=0.25 four of six
factors are rail-bound. Our response at the time was to widen the density range
from 2.5-4 to 2.0-8.0 to accommodate lambda=0.5 -- patching the symptom.

lambda=1 is simply "revenue minus cost", with **no free parameter**. Any
lambda != 1 is not profit, it is an objective function we invented. So
cost_weight is pinned at 1.0 and changing it raises (scan scripts must pass
strict=False explicitly). Price shocks are carried by energy_shock instead; see
below.

Honest statement
----------------
The prices and engineering coefficients below come from public sources, each
cited in place. They are *representative values for European markets and
equipment*, not measurements from one particular greenhouse.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

# ── Physical constants ──────────────────────────────────────
RHO_CP_AIR = 0.335   # Wh/(m^3.K): air 1.2 kg/m^3 x 1005 J/(kg.K)
RHO_CO2 = 1.83       # kg/m^3: CO2 at 20 C, 1 atm (44 g/mol / 24.1 L/mol)


@dataclass
class EconomicModel:
    """Prices and engineering coefficients from public sources; see paper/refs.bib.

    An earlier version used "plausible example values". They were replaced with
    traceable numbers because the Costly axis is denominated in euros, and an
    uncited coefficient makes its absolute value uninterpretable. Two caveats
    remain:
      * these are representative values for European markets and equipment, not
        measurements from one greenhouse;
      * the outdoor temperature setting corresponds to the Mediterranean /
        subtropical winter of the TOMGRO calibration sites (Gainesville /
        Avignon), not to the Netherlands. Dutch greenhouses report 47-69
        m^3/(m^2.yr) of gas, about 460-670 kWh; this model gives about 245 kWh
        under its setting. The difference is climate, not the coefficients.
    """
    # ── Output ─────────────────────────────────────────────
    # Dutch truss tomato averaged 1.57 EUR/kg in 2024-03, Spanish 1.35 EUR/kg
    # (cherry types run higher, 2.45-3.69). We take the midpoint of the range
    # for ordinary truss tomato.
    price_per_kg_fw: float = 1.50    # EUR/kg fresh weight
    dm_fraction: float = 0.055       # fruit dry-matter fraction: fresh = dry / this

    # ── Electricity and supplemental light ─────────────────
    # Eurostat non-household electricity, EU average 18.85 EUR/100kWh in 2024H1.
    elec_price: float = 0.189        # EUR/kWh
    # DLC horticultural lighting qualified products list (2022): efficacy
    # 1.81-3.69 umol/J, over half at >=2.5, only about 6% at >=3.0. We take 2.5
    # as representative of what over half of current fixtures achieve.
    led_efficacy: float = 2.5        # umol/J
    par_ambient: float = 18.0        # mol/(m^2.d) natural baseline; only the excess is supplemental

    # ── Heating ────────────────────────────────────────────
    # Single glass is about 4.54 W/(m^2.K) (0.8 Btu/hr.ft^2.F); double PE
    # measures 2.9-3.4. We take 4.0, single glass with a thermal screen.
    u_value: float = 4.0             # W/(m^2.K)
    t_out_day: float = 14.0          # C, winter value representative of the calibration sites
    t_out_night: float = 8.0
    # Eurostat non-household natural gas, EU average 0.0605 EUR/kWh in 2024.
    heat_price: float = 0.0605       # EUR/kWh

    # ── CO2 enrichment ─────────────────────────────────────
    # Order of magnitude of purchased liquid CO2. In practice growers mostly use
    # boiler flue gas, which is near-free, but running a boiler in summer purely
    # to make CO2 carries an indirect cost. Pricing it as purchased is
    # conservative.
    co2_price: float = 0.12          # EUR/kg CO2
    # Tomato consumes roughly 2 kg CO2 per kg dry matter, at 3-6 kg dry
    # matter/(m^2.yr); allowing about 50% loss this needs 12-24 kg CO2/(m^2.yr).
    # This coefficient gives 14.6 kg/(m^2.yr) at 775 ppm over 300 days, inside
    # that range.
    co2_leak: float = 1.3e-4         # kg/(m^2.d) per ppm above ambient
    co2_ambient: float = 400.0
    # Crack infiltration with the vents shut, as a fraction of the old constant.
    # A modern Venlo house with closed vents leaks about 0.1-0.5 air changes per
    # hour, far below the dozen-plus while venting, so the fraction is small.
    infiltration_frac: float = 0.2
    # Vent capacity ceiling, m^3/(m^2.d). At roughly 60 air changes per hour, a
    # mean height of 5 m and 12 daylight hours: 60 x 5 x 12 = 3600. Above it the
    # house can no longer hold its temperature -- we currently clip rather than
    # model the resulting heat damage; see the note in co2_cost.
    vent_capacity: float = 3600.0
    # CO2 emission factor of burning gas, kg CO2/kWh. Once the flue gas is
    # recovered this is the greenhouse's free CO2 supply. About 0.20 kg/kWh
    # (standard conversion factor; **a primary citation is still owed here:
    # DEFRA/BEIS or IPCC**).
    gas_co2_factor: float = 0.20

    # ── Per-plant cost ─────────────────────────────────────
    # Why this must exist: without it, density is monotonically beneficial in
    # the model, the optimum always sits on the upper rail, that dimension is
    # trivial for an agent, and the optimum does not move across sites -- prior
    # entropy is crushed to near zero.
    # Grafted tomato transplants: US commercial production cost 0.59 (NC) /
    # 1.25 (PA) dollars per plant, retail 1-3 dollars. We take 1.00 EUR/plant.
    transplant_cost: float = 1.00    # EUR/plant, once per cycle
    # **Derived forwards, not reverse-engineered.** This coefficient's comment
    # used to read "calibrated so that the optimal density lands in 2.5-3.5
    # plants/m^2" -- fixing the conclusion first and the coefficient second,
    # which is exactly the practice this paper argues against. It is now derived
    # from labour:
    #   Dutch greenhouse tomato runs about 10 workers/(ha.yr) (5-8 when
    #     transplants are bought in)
    #     -> 0.001 workers/m^2, at a fully loaded seasonal horticultural cost of
    #        about 20k EUR/(worker.yr)
    #     -> about 20 EUR/(m^2.yr) of total labour, which cross-checks against
    #        "labour is 37% of production cost" at an output of roughly
    #        60 EUR/(m^2.yr).
    #   The part that scales with *plant count* is training, lowering and leaf
    #     removal; harvest scales with fruit count (~yield), not plants, and the
    #     literature usually puts harvest at 40-50% of crop labour -> the
    #     plant-count part is about 10 EUR/(m^2.yr).
    #   At 3 plants/m^2: 10 / 3 / 365 ~= 0.0087 EUR/(plant.d).
    # The old reverse-engineered value was 0.010, 15% away -- which says the
    # number itself was not wrong, its justification was. We use the derivation.
    labour_per_plant_day: float = 0.0087   # EUR/(plant.d)
    # Harvest labour, which scales with *yield*. The other half of the same
    # derivation: about 20 EUR/(m^2.yr) total labour, harvest 40-50% of crop
    # labour -> about 9-10 EUR/(m^2.yr); spread over the model's commercial-scale
    # 50 kg/(m^2.yr) -> about 0.19 EUR/kg.
    # **Like labour_per_plant_day, this chain contains numbers we chose (the wage
    #  level, the harvest share). Both should be labelled as our choices rather
    #  than as literature values.**
    harvest_labour_per_kg: float = 0.19    # EUR/kg marketable fruit
    # Unmarketable fraction. Greenhouse tomato grading loss is usually reported
    # at 5-10%; we take the low end, which is the conservative direction because
    # it makes our yield look higher and closer to commercial figures.
    grade_loss: float = 0.05               # dimensionless

    # lambda: **removed**. See the module docstring. The field survives only so
    # that old scan scripts still run, and any value != 1 requires an explicit
    # strict=False -- we want it to make a noise.
    cost_weight: float = 1.0
    strict: bool = True

    # Energy price shock multiplier. It acts on *electricity and gas* and not on
    # the tomato price, transplants or labour -- which is exactly why the
    # Transfer task is decomposable: an agent that modelled only profit cannot
    # answer it, one that modelled revenue and the cost streams separately can
    # re-solve without running anything.
    # Magnitude: the 2021-2022 European energy crisis. Eurostat non-household
    # gas rose from 0.029-0.042 EUR/kWh in the late 2010s to 0.0867 EUR/kWh in
    # 2022H2, the highest on record; non-household electricity excluding taxes
    # peaked at 0.1987 EUR/kWh in 2022H2, and the all-in price including
    # non-recoverable taxes at 0.2151 EUR/kWh in 2023H1. Both are 2-3x
    # pre-crisis.
    energy_shock: float = 1.0

    def __post_init__(self):
        if self.strict and self.cost_weight != 1.0:
            raise ValueError(
                f"cost_weight={self.cost_weight}: lambda is no longer a task design "
                "parameter. An objective with lambda != 1 is not profit, and it "
                "measurably pushes the optima of density / co2 / night_temp outside "
                "commercial ranges (see the module docstring). For a price shock use "
                "energy_shock, or pass strict=False to say you know what you are doing.")

    # ── Daily cost components, EUR/(m^2.d) ─────────────────
    def lighting_cost(self, par):
        extra = np.clip(np.asarray(par, float) - self.par_ambient, 0, None)  # mol/(m^2.d)
        kwh = extra * 1e6 / self.led_efficacy / 3.6e6
        return kwh * self.elec_price * self.energy_shock

    def transmitted_solar(self):
        """Shortwave radiation transmitted into the house, Wh/(m^2.d).

        Inferred back from the natural light baseline par_ambient. Conversion:
        1 MJ of broadband shortwave ~= 2.1 mol PAR (PAR is about 45-50% of
        broadband energy, 1 J PAR ~= 4.57 umol). **This conversion is still owed
        a primary citation.**
        """
        return self.par_ambient / 2.1 / 3.6 * 1000.0

    def heating_cost(self, t_day, t_night):
        """Quasi-steady-state heating demand including solar gain.

        A past error: the old version computed wh = U.(dT_day.12 + dT_night.12),
        applying HL = A.U.dT over all 24 hours. In greenhouse engineering that
        formula is for *equipment sizing* -- how large a boiler the coldest
        *night* needs, explicitly assuming no solar gain (Cornell / UGA extension
        material). Using it as a 210-day operating-energy formula is the wrong
        formula, and heating is 47.8% of total cost, so the error was
        first-order.

        Measured: daytime heat loss is 0.384 kWh/(m^2.d) against 2.381 of
        transmitted solar -- solar gain is 6.2x the daytime loss. The correct
        answer is that you do not heat during the day at all.

        The standard treatment of operating energy is an energy balance with a
        solar term: Bot (1983), Van Henten (1994), de Zwart/KASPRO (1996),
        Vanthoor et al. (2011); reviewed in Katzin, van Henten & van Mourik
        (2022), Agricultural Systems 198:103388. This model runs on a daily step
        and takes their quasi-steady-state simplification: daytime demand minus
        transmitted shortwave, floored at zero.

        Ventilation is not priced here (natural venting through Venlo roof vents
        costs almost no energy), but dumping surplus solar heat requires opening
        the vents, and open vents carry enriched CO2 away. That term lives in
        co2_cost, and it is the real trade-off between day_temp and co2.
        """
        dT_day = np.clip(np.asarray(t_day, float) - self.t_out_day, 0, None)
        dT_night = np.clip(np.asarray(t_night, float) - self.t_out_night, 0, None)
        wh = (np.clip(self.u_value * dT_day * 12.0 - self.transmitted_solar(), 0, None)
              + self.u_value * dT_night * 12.0)                    # W.h/(m^2.d)
        return wh / 1000.0 * self.heat_price * self.energy_shock

    def ventilation_volume(self, t_day):
        """Air exchange needed to dump surplus solar heat, m^3/(m^2.d).

        Whatever daytime shortwave arrives in excess of the loss through the
        cover has to be vented away (Venlo houses vent naturally through roof
        vents, which costs almost no energy, so none is charged here).
        Sensible-heat balance: V = Q_surplus / (rho.c_p.dT), with
        rho.c_p = 1.2 kg/m^3 x 1005 J/(kg.K) = 0.335 Wh/(m^3.K).

        The property that matters: the larger dT, the more heat each cubic metre
        carries away and the less air is needed -- so *a higher day setpoint
        means less venting*. This is what drives the CO2 loss below.
        """
        dT = np.clip(np.asarray(t_day, float) - self.t_out_day, 0.5, None)
        surplus = np.clip(self.transmitted_solar() - self.u_value * dT * 12.0, 0, None)
        v = surplus / (RHO_CP_AIR * dT)
        return np.minimum(v, self.vent_capacity)

    def heating_kwh(self, t_day, t_night):
        """Net heating demand, kWh/(m^2.d). This is heating_cost before pricing."""
        dT_day = np.clip(np.asarray(t_day, float) - self.t_out_day, 0, None)
        dT_night = np.clip(np.asarray(t_night, float) - self.t_out_night, 0, None)
        wh = (np.clip(self.u_value * dT_day * 12.0 - self.transmitted_solar(), 0, None)
              + self.u_value * dT_night * 12.0)
        return wh / 1000.0

    def co2_cost(self, co2, t_day=None, t_night=None):
        """Cost of CO2 enrichment.

        What was wrong before: the old version was extra x co2_leak x price with
        co2_leak a constant whose comment openly said it was "calibrated to land
        inside AHDB's 12-24 range at 775 ppm over 300 days" -- another
        reverse-engineered coefficient. More importantly it was physically wrong:
        the dominant loss path for greenhouse CO2 is *ventilation*, and vent
        volume varies with the day setpoint and the site's climate. It is not a
        constant.

        Commercial practice agrees: with the vents wide open growers reduce or
        stop enriching. This is also what creates a real trade-off between
        day_temp and co2 -- a higher day setpoint means less venting, which makes
        enrichment worth doing; and since vent volume depends on
        dT = T_day - T_out, *the optimal CO2 moves with the site's climate*.

        Boiler flue gas: counting only ventilation loss makes enrichment
        unprofitable at every site and collapses the CO2 factor outright
        (measured optimum 393 ppm, i.e. no enrichment). That is wrong, because it
        omits something real: greenhouse CO2 comes mostly from boiler flue gas as
        a *by-product* of heating, and does not have to be bought. Burning gas
        yields about 0.20 kg CO2/kWh, so the free supply is proportional to the
        heating demand -- cold sites heat more, get more free CO2 and vent less,
        so enrichment is cheap; hot sites heat less, must buy, and vent more, so
        enrichment is expensive. That is precisely the real pattern of heavy
        enrichment in the Netherlands and almost none in the Mediterranean. Only
        the part beyond the flue-gas supply is charged at the purchase price.
        """
        extra = np.clip(np.asarray(co2, float) - self.co2_ambient, 0, None)
        if t_day is None:                       # old call sites: fall back to constant leakage
            return extra * self.co2_leak * self.co2_price
        need = (self.ventilation_volume(t_day) * extra * 1e-6 * RHO_CO2
                + extra * self.co2_leak * self.infiltration_frac)
        free = 0.0
        if t_night is not None:
            free = self.heating_kwh(t_day, t_night) * self.gas_co2_factor
        return np.clip(need - free, 0, None) * self.co2_price

    def plant_cost(self, density, days):
        """Per-plant cost spread over m^2 and day: transplants (once per cycle)
        plus the labour that scales with plant count (every day)."""
        d = np.asarray(density, float)
        return d * self.labour_per_plant_day + d * self.transplant_cost / np.asarray(days, float)

    def energy_cost(self, t_day, t_night, par):
        """The part of cost that scales with energy_shock (electricity + gas)."""
        return self.lighting_cost(par) + self.heating_cost(t_day, t_night)

    def other_cost(self, co2, t_day=None, t_night=None, density=None, days=None):
        """The part that does not scale with energy_shock (purchased CO2,
        transplants, labour).

        t_day enters through the CO2 term: vent volume sets the enrichment loss,
        see co2_cost.
        """
        c = self.co2_cost(co2, t_day, t_night)
        if density is not None and days is not None:
            c = c + self.plant_cost(density, days)
        return c

    def daily_cost(self, t_day, t_night, co2, par, density=None, days=None):
        return (self.energy_cost(t_day, t_night, par)
                + self.other_cost(co2, t_day, t_night, density, days))

    # ── Gross margin (**not net profit**) ──────────────────
    def marketable_kg(self, wm_dry_g_per_m2):
        """Marketable fresh fruit, kg/m^2: gross yield less grading loss.

        grade_loss covers unmarketable fruit (blossom-end rot, cracking,
        off-grade). The old version priced *every* mature fruit at full price
        with no loss at all -- so our "gross yield" was compared directly against
        published *marketable* figures and ran systematically high.
        """
        fw = np.asarray(wm_dry_g_per_m2, float) / self.dm_fraction / 1000.0
        return fw * (1.0 - self.grade_loss)

    def revenue(self, wm_dry_g_per_m2):
        """Revenue from marketable fruit, net of the harvest labour it costs.

        Harvest scales with fruit count (~yield), not with plant count. A
        previous version cut crop labour in half, kept only the plant-count part
        (labour_per_plant_day), and lost the other half entirely -- an omission we
        created, not a simplification. It is added back here.
        """
        kg = self.marketable_kg(wm_dry_g_per_m2)
        return kg * (self.price_per_kg_fw - self.harvest_labour_per_kg)

    def gross_margin(self, wm, t_day, t_night, co2, par, days, density=None):
        """Gross margin per cycle, EUR/m^2 = marketable revenue - the variable
        costs below.

        **This is not net profit.** Not included: depreciation and facility
        capital (glass, structure, heating and CO2 equipment), land or rent,
        water, fertiliser and substrate, crop protection, packaging, transport
        and sales commission, management and overhead, ventilation and cooling
        energy. These are essentially constant in the management factors, so they
        do not move the optimum and cancel out of both R* and the noise -- but
        calling the result net_profit would be wrong.
        """
        return self.revenue(wm) - days * self.daily_cost(
            t_day, t_night, co2, par, density, days)

    # Backwards compatibility: the old name survives, points at the same
    # implementation, and no longer multiplies by cost_weight.
    net_profit = gross_margin

    def profit_rate(self, wm, t_day, t_night, co2, par, days, density=None):
        """Margin **per square metre per day**, EUR/(m^2.d) -- what the benchmark
        should use.

        Using margin per cycle as the objective is wrong: a longer cycle yields
        more while facility occupancy is charged by the day, so the optimal cycle
        length always sits on the upper rail. A real greenhouse maximises return
        per unit area per unit time -- occupying the space has an opportunity
        cost, and the next crop is queueing. As a rate, cycle length acquires an
        interior optimum by itself.
        """
        days = np.asarray(days, float)
        return self.net_profit(wm, t_day, t_night, co2, par, days, density) / days


# ── A site is a place, not just a plant ─────────────────────
# Why this must exist: P_Theta used to vary crop parameters only, while at
# lambda=1 the objective is cost-dominated and cost was identical at every site
# -- so every site wanted the same recipe, one fixed recommendation was nearly
# optimal, and experimenting was worth almost nothing. Measured R*(none): crop
# parameters alone 0.00089, with the site differences below 0.00218. The low
# ceiling was not a shortage of factors, it was agreement between sites.
#
# What real greenhouses actually differ in is climate and prices: a grower in
# Almeria and a grower in the Westland need different recipes and neither can
# copy the other's answer. Every range below has a public source.
SITE_ECON_SPREAD = {
    # Eurostat nrg_pc_205, non-household (500-2000 MWh/yr), 2025H2: lowest
    # Finland 0.0748, highest Ireland 0.2552, EU mean 0.1837 EUR/kWh.
    "elec_price":      (0.0748, 0.2552),
    # Eurostat nrg_pc_203, non-household (10 000-100 000 GJ/yr), 2025H2: lowest
    # Bulgaria 0.0414, highest Sweden 0.1065, EU mean 0.0605 EUR/kWh.
    "heat_price":      (0.0414, 0.1065),
    # DLC horticultural lighting qualified products list (2022), efficacy range.
    "led_efficacy":    (1.81, 3.69),
    # Single glass 4.54 W/(m^2.K) (0.8 Btu/hr.ft^2.F) to measured double PE 2.9.
    "u_value":         (2.9, 4.54),
    # **The weakest-sourced of the six.** The citation we have is two national
    # annual points: Dutch truss tomato 1.57 and Spanish 1.35 EUR/kg in 2024-03.
    # The range widens those two slightly. Real within-season variation is far
    # larger, but we have no series, so we take it narrow -- narrow means sites
    # agree more and the ceiling is lower, which is the direction against us,
    # hence conservative.
    "price_per_kg_fw": (1.30, 1.62),
}

# Climate cannot be drawn independently: a cold place is cold by day and by
# night. A single dimension u in [0,1] walks from a north-west European winter to
# a Mediterranean one, moving both temperatures together.
# Endpoints: the Netherlands (De Bilt) winter mean high/low about 6/1 C; south-
# east Spain (Almeria) about 18/8 C. The TOMGRO calibration sites Gainesville /
# Avignon at 14/8 fall inside the range.
_CLIMATE = {"t_out_day": (6.0, 18.0), "t_out_night": (1.0, 9.0)}


def sample_site_econ(seed: int, **overrides) -> "EconomicModel":
    """Draw one *place*: its climate and its prices. Ranges in SITE_ECON_SPREAD
    and _CLIMATE.

    Uses a different seed offset from the crop parameters
    (sample_instance_params), and the two are independent: we have no evidence
    that Spanish tomato cultivars have systematically different fruit-set
    parameters from Dutch ones.
    """
    r = np.random.default_rng(70_000 + int(seed))
    kw = {k: float(r.uniform(*v)) for k, v in SITE_ECON_SPREAD.items()}
    u = float(r.random())
    for k, (lo, hi) in _CLIMATE.items():
        kw[k] = lo + u * (hi - lo)
    kw.update(overrides)
    return EconomicModel(**kw)


class TomgroProfitModel:
    """TOMGRO plus the economic model, wrapped in the common ground-truth
    backend interface.

    Swappable with zero changes on the environment side: __call__ / oracle /
    response_sd. The response variable is *gross margin in EUR/m^2* rather than
    yield, which is what gives the resource factors an interior optimum.
    """

    def __init__(self, crop, econ: "EconomicModel | None" = None):
        self.crop = crop
        self.econ = econ or EconomicModel()
        self.factors = crop.factors
        self.d = crop.d
        self._oracle = None

    def _unpack(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        f = self.factors
        return [f[i].low + X[:, i] * (f[i].high - f[i].low) for i in range(self.d)]

    def __call__(self, X):
        td, tn, co2, par = self._unpack(X)
        wm = self.crop(X)
        return self.econ.net_profit(wm, td, tn, co2, par, self.crop.days)

    def oracle(self, n_sample: int = 30_000, seed: int = 0):
        if self._oracle is None:
            rng = np.random.default_rng(seed)
            Xs = rng.random((n_sample, self.d)); y = self(Xs)
            i = int(np.argmax(y)); x0, best = Xs[i], float(y[i])
            for r in (0.08, 0.02):
                Xl = np.clip(x0 + rng.normal(0, r, (4000, self.d)), 0, 1)
                yl = self(Xl); j = int(np.argmax(yl))
                if yl[j] > best: x0, best = Xl[j], float(yl[j])
            self._oracle = (x0, best)
        return self._oracle

    @property
    def response_sd(self) -> float:
        rng = np.random.default_rng(12345)
        return float(np.std(self(rng.random((3000, self.d)))))
