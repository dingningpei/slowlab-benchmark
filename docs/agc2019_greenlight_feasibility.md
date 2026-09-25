# AGC 2019 GreenLight feasibility and identifiability audit

## Decision

Phase -1 has reached the limit supported by the public data. The frozen v7 replay passes the pooled calibration-period gate for air temperature, relative humidity, and CO2, but it fails the pre-registered requirement that every compartment pass. The holdout beginning 2020-04-01 remains unopened.

| Scope | Air temperature RMSE | RH RMSE | CO2 RMSE | Gate |
| --- | ---: | ---: | ---: | --- |
| Pooled | 2.035 °C | 4.673 percentage points | 254.9 ppm | Pass |
| AICU | 1.390 °C | 3.392 percentage points | 346.6 ppm | Pass |
| Automatoes | 2.252 °C | 4.053 percentage points | 329.7 ppm | Fail: temperature |
| Digilog | 2.200 °C | 6.023 percentage points | 220.7 ppm | Fail: temperature |
| IUACAAS | 2.417 °C | 4.909 percentage points | 134.2 ppm | Fail: temperature |
| Reference | 2.064 °C | 4.906 percentage points | 224.6 ppm | Fail: temperature |
| TheAutomators | 1.585 °C | 3.994 percentage points | 251.3 ppm | Pass |

The frozen limits are 2.04 °C, 8.5 percentage points, and 361 ppm. All 21 sequences completed, all 1,200 hourly samples were physically valid, and the observed-pipe demand and bounded tracking inputs remained within the published 180/30 W m⁻² rail/grow capacities.

## What the public data determine

The official version-2 archive contains, for each of six compartments, crop parameters, greenhouse climate, root-zone sensor data, laboratory analysis, production, resource use, and tomato quality. Shared files provide weather, documentation, and economics. The accompanying study reports 96 m² floor area, 76.8 m² crop area, roof ventilation with insect netting, two screens, heating capacities, lamps, fogging, and CO2 equipment.

These sources are sufficient to replay observed weather and major realized controls and to compare hourly indoor climate trajectories. The v7 result is therefore evidence for a limited pooled, calibration-period climate-replay claim under observed actions.

## What the public data do not determine

A later WUR report for 96 m² compartments at the same Bleiswijk facility reports a 10 × 9.6 m footprint, seven gutters, a south exterior facade, east/west corridors, and a north neighboring compartment. The 2020 challenge paper maps The Automators, AICU, Reference, IUACAAS, Digilog, and Automatoes to physical compartments 301 through 306. This constrains the footprint and boundary topology, but the later report cannot certify that all 2019 materials and thermal conditions were unchanged.

The archive and studies still do not report the parameters needed to reproduce each physical compartment: the 2019 roof slope and air-zone heights, corridor and neighboring-compartment boundary temperatures, structure/floor heat capacity, pipe geometry, screen heat exchange, pressure coefficients, or insect-net resistance. They also do not uniquely identify lamp heat partition, the conflicting floor/crop/growing-area denominator, or each team's initial crop state.

The residual timing supports the same diagnosis. Pooled air-temperature RMSE is 1.802 °C with lights on and 2.545 °C with lights off. Excluding the first 24 hours of each short sequence as a diagnostic lowers pooled RMSE to 1.858 °C, but Automatoes and IUACAAS remain above the threshold at 2.145 and 2.108 °C. The frozen score retains those first-day samples; the diagnostic cannot be used as a post-hoc pass.

This is material rather than cosmetic. In v7, AICU has a +0.084 °C temperature bias, while Automatoes, Digilog, IUACAAS, and Reference have biases of −1.721, −1.322, −1.700, and −1.124 °C. A single global heat correction would move the already passing AICU in the wrong direction. Team-specific offsets could reduce calibration error, but they would encode the outcomes being validated and would not establish physical fidelity.

## Scientific boundary

No further scalar tuning should be selected from these calibration outcomes. The holdout stays closed. Strict per-compartment validation can resume only if independent compartment drawings/specifications or system-identification measurements determine the missing parameters before the holdout is read.

Until then, the paper may state that the simulator reproduces the pooled calibration-period climate trajectory at the published GreenLight error scale under observed actions. It may not call this per-compartment validation, holdout validation, counterfactual actuator validation, or an AGC digital twin. Observed-pipe replay is a historical boundary condition and cannot supply the heating response for new counterfactual policies.

Primary facility sources are the [2019/2020 challenge study](https://doi.org/10.3390/s20226430) and [WUR Report WPR-1329](https://edepot.wur.nl/661490). Machine-readable evidence is in `configs/agc2019_calibration_result_v7.json` and `configs/agc2019_identifiability_audit_v0.json`.
