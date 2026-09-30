# Version 2.2 reality-constraint contract

## Decision

Phase 0 is complete and the frozen GreenLight v8 external trajectory gate failed. Across
eight sequences, 137 compartment-days and 3,288 hourly samples, pooled temperature RMSE
was 2.579 °C against a 2.04 °C limit, relative-humidity RMSE was 9.983 percentage points
against an 8.5-point limit, and CO2 RMSE was 211.8 ppm against a 361 ppm limit. Passing
CO2 alone does not satisfy the every-metric, every-compartment rule.

Version 2.2 therefore proceeds as a **reality-constrained simulation benchmark**. The
failure is an external-validity result, not a tuning target. V8 will not be refit on the
opened holdout and the first paper will not claim a validated digital twin, a validated
real-greenhouse closed loop, or arbitrary counterfactual validity.

The machine-readable contract is
[`configs/reality_constraints_v2_2.json`](../configs/reality_constraints_v2_2.json).
It distinguishes four evidence classes:

| Class | Meaning in Version 2.2 |
| --- | --- |
| `observed` | Directly present in a public time series or measurement table. The range is descriptive support, not causal validation. |
| `literature_constrained` | Reported equipment, geometry or response range without an aligned realised trajectory. |
| `assumed` | A transparent benchmark choice requiring sensitivity analysis. |
| `unsupported` | Not measured or identified by the public sources; ineligible for the formal reality-constrained matrix. |

## Data roles

- **AGC 2019** supplies development evidence and the already opened failed holdout audit.
  It cannot become prospective validation for Version 2.2.
- **AGC 2023** supplies observed climate/action envelopes and crop-state diagnostics. It
  contains no realised CO2 dosing or fogging input and is development-only.
- **AGC 2024** supplies realised-action schemas, observed input envelopes and reported
  facility capacities. It does not directly identify counterfactual actuator responses.
- **Root-zone literature** supplies candidate envelopes and sensitivity checks. The
  dynamic irrigation/EC model remains a Phase 1 decision rather than a frozen core factor.

## What remains unsupported

The public records do not provide realised fogging flow, heating-water mass flow,
facility-specific natural-ventilation airflow, or screen heat/moisture flux. Installed
capacity is not a realised trajectory. These quantities may appear only in a clearly
labelled, predeclared sensitivity analysis until independent evidence is available.
The archives also do not identify facility sensor-error distributions, command-to-actuator
delay distributions or labelled actuator failure rates. Version 2.2 must describe any such
models as assumptions and test them separately rather than presenting them as calibrated.

AGC 2024 resource columns also require care: reported heating energy is derived partly
from indoor temperature, lighting electricity is calculated from lamp activation and an
assumed efficacy, and CO2 mass is reconstructed from cumulative dosing minutes. They are
not independent physical-flux sensors.

## Enforcement

Every Version 2.2 value covered by a reality claim must carry a `constraint_id`. Run:

```bash
python3 scripts/audit_reality_support.py proposed_records.json
```

The audit follows [`configs/reality_audit_policy_v1.json`](../configs/reality_audit_policy_v1.json),
which supersedes the manifest's original out-of-range rule. Each record falls into exactly
one category:

| Category | Meaning | Fails |
| --- | --- | --- |
| within support | inside the observed or literature envelope | no |
| source extrapolation | physically feasible, outside the evidence envelope; counted per site | no |
| declared assumption | assumed or unsupported quantity whose value is set by a named frozen file (`declared_in`), in an allowed analysis role | no |
| physical error | violates a physical law, a declared equipment capacity or the pinned model's validity domain, or is not finite | yes |
| undeclared assumption | assumption-type quantity without an existing declaring file, or a sensitivity-only quantity (executor delay, executor failure rate, fogging) used in the main analysis | yes |
| unknown constraint | no such `constraint_id` | yes |

An evidence envelope describes data coverage, not a physical limit. The audit never removes,
clips or reweights a record or a site; extrapolation is disclosed instead. Passing
establishes neither causal nor counterfactual validity.

The frozen Phase 0 conclusion can be checked without the external raw archives:

```bash
python3 scripts/verify_phase0_reality_contract.py
```

This command verifies the committed result against the committed gate and constraint
contract. Re-running the complete trajectory simulation still requires the public raw
data and the frozen replay manifest.
