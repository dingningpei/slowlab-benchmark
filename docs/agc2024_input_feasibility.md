# AGC 2024 input-only feasibility audit

> Status (2026-09-29): the replay route assessed here was not pursued; the benchmark is a reality-constrained simulation, not an AGC replay. This audit remains the source of the facility capacities (heating, CO2, lighting) used in the task contract and is cited by `configs/reality_constraints_v2_2.json`.

## Decision

The public AGC 2024 dwarf-tomato archive is substantially better suited than AGC 2019 to a prospective observed-action climate replay. It receives a provisional **B** rating: all six compartments expose nearly complete five-minute weather and realised equipment-state traces, and the accompanying paper reports much of the equipment and geometry needed to construct a source-grounded replay. A model, reconstruction rules, uncertainty bounds, identities and pass/fail gate must still be frozen before indoor temperature, humidity or CO2 values are read.

The archive receives a **C** rating for direct counterfactual actuator validation. It does not independently measure natural ventilation flow, screen heat/moisture flux, crop transpiration or delivered heating flux. Those relationships would still be supplied by a model rather than verified by the archive.

No indoor temperature, relative-humidity, humidity-deficit or CO2 concentration value was directly accessed in this audit, and no validation identity or period has been selected. The archive schema names those columns, but the auditor does not index their values.

## Input coverage

The archive SHA-256 is `19bfddf994ffeeb6eff8c20da14af8ae09956d23d7c6d85d7e5aa017cff3f7b1`. It contains six compartment CSVs, a common weather trace, a weather forecast, a data dictionary and harvest measurements. The camera archive is separate and was not downloaded.

| Compartment/team | Five-minute rows | Essential realised-input complete rows | Period |
| --- | ---: | ---: | --- |
| Agrifusion | 22,621 | 22,608 | 2024-09-03 to 2024-11-20 |
| IDEAS | 20,317 | 20,304 | 2024-09-03 to 2024-11-12 |
| MuGrow | 20,029 | 20,016 | 2024-09-03 to 2024-11-11 |
| Reference | 21,181 | 21,168 | 2024-09-03 to 2024-11-15 |
| Tomatonuts | 22,909 | 22,896 | 2024-09-03 to 2024-11-21 |
| Trigger | 22,909 | 22,896 | 2024-09-03 to 2024-11-21 |

Every compartment contains realised lower-pipe temperature, lee/wind vent positions, energy and blackout screen positions, lamp activation, CO2 actuation state and cumulative dosing minutes. Each essential channel is approximately 99.94% complete. The common weather trace has 23,039 rows with complete outside temperature, RH, global radiation, wind speed/direction, rain and pyrgeometer heat-emission channels.

The 13 incomplete realised-input rows are the shared initial 00:00 row plus the twelve repeated local-clock records during the 2024-10-27 daylight-saving transition. This is an input-only finding. A later frozen sequence constructor can split at the transition or operate in UTC; it must not select periods using indoor outcomes.

Plant density is an event parameter rather than a five-minute measurement. The paper fixes the initial density at 56 plants/m2 and permits changes to 42, 30 and 20 plants/m2, so its sparse observations should be interpreted as change events and forward-filled under a rule frozen before validation.

## What is source-grounded

Maree et al. (2025) report:

- a 10 m by 9.6 m, 96 m2 compartment and a cross-section in Figure A1;
- one exterior side wall and one wall shared with another compartment;
- a single floor pipe-rail circuit with 120 W/m2 peak capacity;
- continuous roof ventilation with 0.3 m2 opening per m2 floor and a 0.40 by 0.45 mm anti-insect net;
- LUXOUS 1547 D FR energy and OBSCURA 9950 FR W blackout screens;
- eight Fluence VYPR 4i B9F WB dimmable LED fixtures, 20-200 micromol m-2 s-1, with an assumed efficacy of 3.2 micromol/J;
- fogging capacity of 220 g m-2 h-1 and CO2 capacity of 7.5 g m-2 h-1;
- Pick-&-Joy Red Cherry dwarf tomato in 13 cm pots on three 1.62 by 6.12 m, 0.8 m-high tables.

This removes several ambiguities encountered in AGC 2019, although it does not by itself identify all effective heat-transfer, screen, insect-net, leakage and crop-transpiration parameters.

## What the resource columns mean

The archive's resource fields must not be mistaken for independent physical-flux sensors:

- heating energy is calculated from `2 * max(0, pipe temperature - indoor air temperature)` W/m2; it is not a heat-meter observation;
- lighting electricity is calculated from lamp activation, a 200 micromol m-2 s-1 maximum and assumed 3.2 micromol/J efficacy;
- CO2 mass is calculated from changes in cumulative dosing minutes using 0.125 g m-2 min-1.

An initial auditor summarized resource-field ranges because the precommitted protocol allowed resource inputs. Once the paper revealed that heating energy embeds the forbidden indoor-temperature outcome, no identity or period had been selected. The final auditor excludes all energy and economics values and records only their column names. This disclosure preserves the blind-test boundary.

## Remaining gaps

The public timeseries does not contain:

- heating-water mass flow and separate supply/return temperatures;
- measured natural-ventilation airflow or a compartment-specific discharge calibration;
- a realised fogging state or fog water-flow channel;
- screen heat/moisture fluxes and effective material coefficients;
- continuous LAI or canopy transpiration;
- the parameterized Kaspro compartment model or dwarf-tomato crop model.

The last point is important: the paper states that teams were given Kaspro parameterized for these compartments and a dwarf-tomato crop model. This demonstrates that relevant facility/crop modelling assets existed for the experiment, but they are not included in the public timeseries archive. Their availability and licensing should be asked of the authors rather than assumed.

## Next scientific step

Keep every 2024 indoor climate value closed. Use AGC 2019 and the public 2023 pre-trial as development data to define the v9 physical/controller reconstruction, including explicit treatment of fogging and crop state. In parallel, ask the AGC authors for the parameterized Kaspro configuration, dwarf-tomato crop model, realised fogging channel if retained, heating-system metadata and sensor calibration/location metadata.

Only after that development work should a separate contract freeze the 2024 compartment-period identities, UTC/DST handling, permitted initial-state observation, model and source hashes, uncertainty ensemble, metrics, thresholds and one-shot rule. Passing that future test would establish observed-action trajectory transfer. It would still not by itself validate arbitrary counterfactual policies.

Sources: [AGC 2024 dataset](https://doi.org/10.4121/fa102772-32db-4b30-bace-12f2016722ce), [Maree et al. 2025](https://doi.org/10.3390/s25144321).
