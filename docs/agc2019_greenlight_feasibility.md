# AGC 2019 GreenLight feasibility and identifiability audit

## Decision

Phase -1 completed its pre-registered one-shot temporal holdout. The frozen v8 hybrid replay **does not pass** the required “与真实温室轨迹一致” gate. All eight sequences completed without sampled physical violations, and pooled CO2 passed, but pooled temperature and relative humidity failed. The rule also required every compartment to pass.

| Scope | Air temperature RMSE | RH RMSE | CO2 RMSE | Gate |
| --- | ---: | ---: | ---: | --- |
| Pooled | 2.579 °C | 9.983 percentage points | 211.8 ppm | Fail: temperature, RH |
| AICU | 2.035 °C | 11.049 percentage points | 116.8 ppm | Fail: RH |
| Automatoes | 4.127 °C | 6.219 percentage points | 444.1 ppm | Fail: temperature, CO2 |
| Digilog | 4.033 °C | 7.132 percentage points | 187.5 ppm | Fail: temperature |
| IUACAAS | 1.837 °C | 8.916 percentage points | 82.7 ppm | Fail: RH |
| Reference | 1.965 °C | 8.932 percentage points | 190.2 ppm | Fail: RH |
| TheAutomators | 2.416 °C | 15.752 percentage points | 189.0 ppm | Fail: temperature, RH |

The frozen limits are 2.04 °C, 8.5 percentage points, and 361 ppm. The holdout contains 3,288 hourly samples over 137 compartment-days beginning after 2020-04-01. This failure cannot be repaired by refitting v8 on the opened holdout. A scientifically valid continuation must define a new prospective model version and evaluate it on a new untouched temporal, facility, or experimental holdout.

The residual layer improves calibration fit, but its temporal transfer is insufficient and it does not validate latent crop dynamics or counterfactual actuator responses. The current simulator therefore cannot support the paper's desired real-greenhouse trajectory-consistency claim. The exact frozen outcome is in `configs/agc2019_holdout_result_v8.json`.

## What the public data determine

The official version-2 archive contains, for each of six compartments, crop parameters, greenhouse climate, root-zone sensor data, laboratory analysis, production, resource use, and tomato quality. Shared files provide weather, documentation, and economics. The accompanying study reports 96 m² floor area, 76.8 m² crop area, roof ventilation with insect netting, two screens, heating capacities, lamps, fogging, and CO2 equipment.

These sources are sufficient to replay observed weather and major realized controls and to compare hourly indoor climate trajectories. The v7 result is therefore evidence for a limited pooled, calibration-period climate-replay claim under observed actions.

## What the public data do not determine

A later WUR report for 96 m² compartments at the same Bleiswijk facility reports a 10 × 9.6 m footprint, seven gutters, a south exterior facade, east/west corridors, and a north neighboring compartment. The 2020 challenge paper maps The Automators, AICU, Reference, IUACAAS, Digilog, and Automatoes to physical compartments 301 through 306. This constrains the footprint and boundary topology, but the later report cannot certify that all 2019 materials and thermal conditions were unchanged.

The archive and studies still do not report the parameters needed to reproduce each physical compartment: the 2019 roof slope and air-zone heights, corridor and neighboring-compartment boundary temperatures, structure/floor heat capacity, pipe geometry, screen heat exchange, pressure coefficients, or insect-net resistance. They also do not uniquely identify lamp heat partition, the conflicting floor/crop/growing-area denominator, or each team's initial crop state.

The residual timing supports the same diagnosis. Pooled air-temperature RMSE is 1.802 °C with lights on and 2.545 °C with lights off. Excluding the first 24 hours of each short sequence as a diagnostic lowers pooled RMSE to 1.858 °C, but Automatoes and IUACAAS remain above the threshold at 2.145 and 2.108 °C. The frozen score retains those first-day samples; the diagnostic cannot be used as a post-hoc pass.

This is material rather than cosmetic. In v7, AICU has a +0.084 °C temperature bias, while Automatoes, Digilog, IUACAAS, and Reference have biases of −1.721, −1.322, −1.700, and −1.124 °C. A single global heat correction would move the already passing AICU in the wrong direction. Team-specific offsets could reduce calibration error, but they would encode the outcomes being validated and would not establish physical fidelity.

## Scientific boundary after the holdout

Do not report v8 as passing real-trajectory validation, and do not tune v8 against this holdout. The calibration and holdout together are useful diagnostic evidence: the remaining errors are compartment- and period-dependent, especially the cold bias in Automatoes and Digilog and the humid bias in AICU and TheAutomators. They point to missing compartment boundary conditions, moisture processes, and actuator/system-identification data rather than a defensible global scalar correction.

The next version should be treated as a new prospective study. It needs an explicit new validation source that remains unopened during development. The strongest route is independent greenhouse trajectory data with weather, realized heating, ventilation, screening, fogging/humidification, CO2, lighting, indoor temperature, RH/VPD, and CO2 at aligned timestamps, plus facility geometry and actuator capacities. If those data cannot be obtained, narrow the paper's claim to a synthetic benchmark and present AGC replay as a failed external-validity audit rather than evidence of a digital twin.

Primary facility sources are the [2019/2020 challenge study](https://doi.org/10.3390/s20226430) and [WUR Report WPR-1329](https://edepot.wur.nl/661490). Machine-readable evidence is in `configs/agc2019_calibration_result_v7.json`, `configs/agc2019_calibration_result_v8.json`, `configs/agc2019_identifiability_audit_v0.json`, and `configs/agc2019_holdout_result_v8.json`.
