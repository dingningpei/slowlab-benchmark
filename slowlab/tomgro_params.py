"""Published parameter values for the reduced TOMGRO model.

Two independent sources; symbols follow the originals and are not renamed:

[VC14] Vazquez-Cruz, M.A., et al. (2014). Global sensitivity analysis by means of
       EFAST and Sobol' methods and calibration of reduced state-variable TOMGRO
       model using genetic algorithms. Computers and Electronics in Agriculture 100, 1–12.
       -> Table 3: *nominal values* for 17 parameters with +/-10% bounds, under greenhouse experimental conditions

[G21]  Gong, L., Yu, M., Jiang, S., et al. (2021). Studies of evolutionary algorithms
       for the reduced Tomgro model calibration for modelling tomato yields.
       Smart Agricultural Technology 1, 100011.
       -> Table 3: *calibration search intervals* for 14 parameters

The two are different kinds of number: VC14 gives nominal point estimates
(usable as model defaults), G21 gives the search ranges used during calibration
(wider, and suitable as sampling intervals for plant-to-plant perturbation).
"""
from __future__ import annotations

# ── VC14 Table 3: nominal values plus +/-10% bounds ────────────────────
# key: (nominal, lower, upper, unit, description)
VC14_NOMINAL: dict[str, tuple] = {
    "Nm":   (0.495,  0.4455,  0.5445,  "node/d",         "maximum rate of node appearance"),
    "delta":(0.041,  0.0369,  0.0451,  "m^2/node",       "maximum leaf area expansion per node"),
    "beta": (0.22,   0.198,   0.242,   "1/node",         "coefficient of the expolinear equation"),
    "Nb":   (18.5,   16.65,   20.35,   "node",           "projection of the linear LAI-N segment on the abscissa"),
    "E":    (0.7,    0.63,    0.77,    "1/d",            "growth efficiency (biomass per available photosynthate)"),
    "D":    (0.108,  0.0972,  0.1188,  "g/(m^2.h)",      "CO2 to CH2O conversion coefficient"),
    "sigma":(0.0664, 0.05976, 0.07304, "umol/(m^2.s)",   "CO2 conductance"),
    "K":    (0.58,   0.522,   0.638,   "node",           "development time from first fruit to first mature fruit"),
    "m":    (0.1,    0.09,    0.11,    "-",              "light transmission coefficient"),
    "Km":   (0.006,  0.0054,  0.0066,  "g/(g.d)",        "maintenance respiration coefficient"),
    "fcN":  (0.85,   0.765,   0.935,   "- (0-1)",        "biomass partitioning coefficient by development stage"),
    "uh":   (30.0,   27.0,    33.0,    "umol CO2/(m^2.s)","photosynthesis parameter (upper)"),
    "ul":   (5.0,    4.5,     5.5,     "umol CO2/(m^2.s)","photosynthesis reduction parameter (lower)"),
    "alphaF":(0.95,  0.855,   1.045,   "1/d",            "maximum partitioning fraction to fruit"),
    "nu":   (0.24,   0.216,   0.264,   "1/node",         "transition from vegetative to reproductive growth"),
    "NFF":  (14.88,  13.392,  16.368,  "node",           "node number at the appearance of the first fruit"),
    "alpha":(0.09,   0.081,   0.099,   "umol/umol",      "radiation use efficiency"),
}

# ── G21 Table 3: calibration search intervals (wider, suitable for plant-to-plant perturbation) ──
G21_SEARCH_RANGE: dict[str, tuple] = {
    "Nm":    (0.35,  0.40,   "node/d"),
    "Nb":    (14.0,  16.0,   "node"),
    "delta": (0.05,  0.08,   "m²/node"),
    "beta":  (0.45,  0.55,   "1/node"),
    "Vmax":  (8.0,   10.0,   "g[d.w.]/node"),
    "tau":   (0.08,  0.12,   "μmol/(m²·s)"),
    "Tcrit": (19.0,  21.0,   "°C"),
    "nu":    (0.8,   1.0,    "1/node"),
    "K":     (0.8,   1.0,    "node"),
    "m":     (0.01,  0.015,  "—"),
    "NFF":   (16.0,  18.0,   "node"),
    "alphaF":(0.8,   1.0,    "1/d"),
    "E":     (0.9,   1.2,    "g[d.w.]/g[CH2O]"),
    "D":     (4.0,   6.0,    "g/(m²·h)"),
}

# ── Where the two sources disagree on the same symbol ─────────────────
# Not an error: they are calibrations at different sites and cultivars. Recorded
# because the disagreement is itself direct evidence of how large the parameter
# uncertainty is -- far wider than any +/-10% sensitivity interval.
DISAGREEMENTS = {
    "Nm":    "VC14 nominal 0.495 vs G21 search interval [0.35,0.40] -- no overlap",
    "delta": "VC14 0.041 vs G21 [0.05,0.08] -- no overlap",
    "beta":  "VC14 0.22  vs G21 [0.45,0.55] -- a factor of two apart",
    "Nb":    "VC14 18.5  vs G21 [14,16]",
    "E":     "VC14 0.7   vs G21 [0.9,1.2]",
    "D":     "VC14 0.108 vs G21 [4,6] -- units may differ; check the originals",
    "nu":    "VC14 0.24  vs G21 [0.8,1.0]",
    "NFF":   "VC14 14.88 vs G21 [16,18]",
    "K":     "VC14 0.58  vs G21 [0.8,1.0]",
    "m":     "VC14 0.1   vs G21 [0.01,0.015] -- an order of magnitude apart; different definition or units",
}

# The two sources share 11 parameters, and *10 of them have non-overlapping
# ranges*. This is not a transcription error, it is two site/cultivar
# calibrations. The implication matters: the claim that a mechanistic model gives
# you external anchoring is qualified -- the parameter uncertainty itself is far
# larger than any single paper's +/-10% sensitivity interval. The paper must say
# which set it used and treat the other as a sensitivity analysis.
N_SHARED, N_DISJOINT = 11, 10

# ── [J99] Jones, Kenig & Vallejos (1999), Trans. ASAE 42(1):255–265 ──
#     The original paper. Equations 1-11 and the measured calibration values of Table 3.
#
# The full equation set, numbered as in the original:
#   (1)  dN/dt   = Nm . fN(T)                                    T is hourly temperature
#   (2)  LAI     = ρ · (δ/β) · ln{1 + exp[β(N−Nb)]}              expolinear, Goudriaan & Monteith 1990
#   (4)  dLAI/dt = rho . delta . lambda(Td) . exp[beta(N-Nb)]/(1+exp[beta(N-Nb)]) . dN/dt   if LAI <= LAImax, else 0
#   (5)  GRnet   = E · (Pg − Rm) · [1 − fR(N)]
#   (6)  Rm      = ∫ Q10^((T−20)/10) · rm · (W − WM) dt
#   (7)  dWF/dt  = GRnet . alphaF . fF(Td) . [1 - e^(-nu(N-NFF))] . g(Tdaytime)   if N > NFF
#   (8)  g(Tdaytime) = 1.0 - 0.154 . (Tdaytime - TCRIT)          if Tdaytime > TCRIT, else 1.0
#   (9)  dW/dt   = GRnet − p1 · ρ · dN/dt
#  (10)  (dW/dt)max = dWF/dt + (Vmax - p1) . rho . dN/dt         actual dW/dt is the smaller of (9) and (10)
#  (11)  dWM/dt = DF(Td) . (WF - WM)                             when N > NFF + kappaF
#
# Parameters carried over from TOMGRO without recalibration (from the original text):
J99_CARRIED_OVER = {
    "Nm": {"Gainesville": 0.50, "LakeCity": 0.50, "Avignon": 0.55},   # node/d
    "p1": 2.0,      # g[leaf]/node, leaf removed per node once LAImax is reached
    "rm": 0.016,    # g[CH2O]/(g[d.w.].d), maintenance respiration coefficient
}

# Table 3: least-squares calibration values. The three columns are Gainesville /
# Lake City / Avignon. A bracketed value means the Gainesville estimate was
# carried over; * means there was too little data to estimate it.
J99_TABLE3 = {
    #          Gainesville  LakeCity   Avignon
    "delta":  (0.038,      0.038,     0.030),    # m^2/node  <- Lake City carried over
    "beta":   (0.169,      0.169,     0.169),    # 1/node
    "Nb":     (16.0,       16.0,      16.0),     # node
    "alphaF": (0.80,       0.95,      0.95),     # bounded above at 0.95
    "vartheta":(0.135,     0.200,     0.200),    # ϑ, 1/node
    "NFF":    (22.0,       10.0,      19.0),     # node
    "TCRIT":  (24.4,       24.4,      22.0),     # C         <- Lake City carried over
    "kappaF": (5.0,        5.0,       None),     # node, node lag from first fruit to maturity
    "DFmax":  (0.08,       0.04,      None),     # 1/d
    "Vmax":   (None,       None,      8.0),      # g[d.w.]/node
}
J99_LAIMAX = {"LakeCity": 4.0, "Avignon": 2.3}   # Gainesville did not reach LAImax in either year
J99_DENSITY = {"Gainesville1993": 3.10, "Gainesville1994": 3.50,
               "Avignon": 2.20, "LakeCity": 3.12}   # ρ, plants/m²

# The only temperature function given in full analytic form (equation 8):
def g_daytime_vec(T_daytime, TCRIT):
    """Vectorised form of equation (8)."""
    import numpy as np
    T = np.asarray(T_daytime, float)
    return np.clip(np.where(T <= TCRIT, 1.0, 1.0 - 0.154 * (T - TCRIT)), 0.0, 1.0)


def g_daytime(T_daytime: float, TCRIT: float) -> float:
    """Suppression function for heat-induced fruit abortion. The slope 0.154 is from Vallejos et al. (1997)."""
    if T_daytime <= TCRIT:
        return 1.0
    return max(0.0, 1.0 - 0.154 * (T_daytime - TCRIT))


# ── Still missing: the explicit form of the temperature response functions ──
# The five state equations are confirmed (both papers agree):
#   dN/dt   = Nm · f_N(T)
#   dLAI/dt = ρ · δ · λ(Td) · exp(β(N−Nb))/(1+exp(β(N−Nb))) · dN/dt
#   dW/dt   = dWF/dt + (Vmax − p1) · ρ · dN/dt
#   dWF/dt  = GRnet · αF · fF(Td) · [1 − exp(−ν(N−NFF))] · g(T_daytime)
#   dWM/dt  = DF(Td) · (WF − WM)
# But for the analytic form of the functions below, both papers merely cite Jones
# et al. (1999). J99 calls them "exactly as in the full TOMGRO" and does not write
# them out, so one has to go a level further back to Jones et al. (1991) and
# Acock et al. (1978).
MISSING_FUNCTIONS = {
    "fN(T)":      "temperature reduction function for node development -> Jones et al. (1991) TOMGRO",
    "lambda(Td)": "temperature reduction function for leaf area expansion -> as above",
    "fF(Td)":     "temperature reduction function for fruit partitioning -> as above",
    "DF(Td)":     "fruit maturation rate as a function of mean daily temperature -> Marcelis & Koning (1995), scaled by DFmax",
    "Pg":         "daily integrated gross photosynthesis -> Acock et al. (1978); Acock (1991)",
    "fR(N)":      "partitioning fraction to roots, varying with development node -> Jones et al. (1991)",
    "Q10":        "temperature coefficient of maintenance respiration (form known, value not given in J99)",
}
SOURCE_FOR_MISSING = ("Jones, J.W., et al. (1991) TOMGRO; "
                      "Acock, B., et al. (1978) canopy photosynthesis; "
                      "Marcelis & Koning (1995) fruit development.")


def nominal_params() -> dict[str, float]:
    """VC14's nominal values, used as the model defaults."""
    return {k: v[0] for k, v in VC14_NOMINAL.items()}


def perturbation_intervals(source: str = "VC14") -> dict[str, tuple[float, float]]:
    """Sampling intervals for plant-to-plant perturbation.

    VC14 gives +/-10% sensitivity intervals (narrow); G21 gives calibration search
    intervals (wide). The two differ considerably in width -- which is itself a
    statement about how large the parameter uncertainty is, so the paper should
    say which one it used.
    """
    if source == "VC14":
        return {k: (v[1], v[2]) for k, v in VC14_NOMINAL.items()}
    if source == "G21":
        return {k: (v[0], v[1]) for k, v in G21_SEARCH_RANGE.items()}
    raise ValueError(f"unknown source {source!r}; use 'VC14' or 'G21'")
