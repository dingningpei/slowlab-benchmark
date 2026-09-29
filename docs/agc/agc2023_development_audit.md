# AGC 2023 pre-trial development audit

## Decision

The public AGC 2023 pre-trial archive is suitable for **v9 development**, especially for temperature/RH observed-action replay and dwarf-tomato crop-state diagnostics. It is not a validation dataset: it represents one compartment, has already been opened for development, and does not expose CO2 dosing or fogging actuation.

The 4.03 MiB time-series archive is sufficient for this stage. The 4.29 GiB canopy-camera and 7.11 GiB single-plant-camera archives were not downloaded. Direct leaf-area measurements are already present, so the images are not required to construct sampled LAI.

## Climate and realised actions

`ClimateTimeseries.xlsx` contains 18,721 five-minute records from 2023-09-05 00:00 through 2023-11-09 00:00 with no cadence breaks. The numeric climate and control channels each have 18,705 usable values (99.9145% coverage); the first 16 records are missing.

| Channel | Mean | Range | Recorded changes |
| --- | ---: | ---: | ---: |
| Indoor temperature (°C) | 21.34 | 16.50–35.20 | 14,910 |
| Indoor RH (%) | 78.91 | 33.10–94.50 | 17,380 |
| Indoor CO2 (ppm) | 478.42 | 349.00–827.00 | 18,081 |
| Rail-pipe temperature (°C) | 6.41 | 0.00–52.10 | 3,872 |
| Lee vent (%) | 30.49 | 0.00–100.00 | 7,414 |
| Energy screen (%) | 57.45 | 0.00–100.00 | 5,002 |
| Lamp state | 0.569 | 0–1 | 452 |

The weather trace, screen positions, lamp state, lee/wind vents and rail-pipe temperature give useful excitation for developing observed-action climate replay. Indoor CO2 is available only as an outcome. Without a dosing input, it cannot identify a CO2 actuator-response model.

## Crop observations and sampled LAI

The destructive-harvest workbook directly measures leaf area in cm²/plant. For each date × light × EC group, sampled LAI is computed as:

\[
\mathrm{LAI}=\frac{\overline{\text{leaf area in cm}^2/\text{plant}}\times\text{plants/m}^2}{10{,}000}.
\]

There are 32 groups: four harvest dates, four light levels, two EC levels. The transplant observation has 12 plants per group; later harvests have six. The resulting group-mean LAI ranges are:

| Date | Trusted density (plants/m²) | Group-mean LAI range |
| --- | ---: | ---: |
| 2023-09-04 | 48 | 0.168–0.168 |
| 2023-09-18 | 48 | 1.124–1.508 |
| 2023-09-29 | 25 | 0.972–1.697 |
| 2023-11-09 | 20 | 2.121–2.987 |

The identical transplant values across treatment labels are expected because the treatments had not yet produced separate histories. These are biological plant replicates inside one compartment, not independent greenhouse sites.

The non-destructive consolidated sheet contains 360 plant-week records on nine dates, with 331 non-missing plant-height observations. It supports crop-state and treatment-response development between destructive harvests.

## Source corrections that must remain explicit

Two workbook issues would silently corrupt the model if untreated:

1. The destructive workbook's combined table records plant density `1050` for every transplant row. This conflicts with the Info sheet and the physical table count: 242 plants on 3.6 × 1.4 m is approximately 48 plants/m². The audit replaces `1050` with 48 only for the sampled transplant date.
2. The Info sheet gives 25 plants/m² for 22–29 September and 20 plants/m² for 9 October–9 November, leaving 29 September–9 October unresolved. Sample-date LAI is identifiable, but a continuous density trajectory across that gap is not. The model must not silently interpolate it.
3. The separate `week 38` and `week 39` tabs in `CropMeasurements.xlsx` contain 2024 date typos. The consolidated `All data` sheet contains the internally consistent 2023 dates and is the frozen source.

## Reproduction

Run the audit with one worker and one numerical-library thread:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/agc/audit_agc2023_development.py \
  --data-dir /path/to/agc2023_timeseries \
  --schema configs/agc/agc2023_input_schema_audit_v0.json \
  --out configs/agc/agc2023_development_summary_v0.json
```

The streaming audit completes in about 1.2 seconds on the current MacBook. It checks all three workbook hashes before reading them.

## Consequence for v9

Use AGC 2023 to develop the climate interface, crop initialization and LAI/transpiration component, with internal blocked or rolling-origin diagnostics if needed. Do not report those diagnostics as external validation. Keep AGC 2024 indoor outcomes closed until the implementation, ensemble, identities, periods, initialization rule, metrics and thresholds are frozen.

Source: [AGC 2023 pre-trial dataset](https://doi.org/10.4121/e1ee9de9-6ce9-4502-a37c-34b5b1372bed).
