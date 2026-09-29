# Original GreenLight implementation-validation preflight

## What is being validated

This stage checks whether the official Python GreenLight implementation can reproduce the Katzin et al. (2020) evaluation workflow and published error scale. It is an implementation check, not evidence that GreenLight is calibrated to an Autonomous Greenhouse Challenge compartment and not validation of arbitrary counterfactual controls.

The implementation is frozen to official GreenLight `v2.0.5`, commit `fa502eddae5f9eff7b3380c88037d9b5f3f14bf5`. Its `main_katzin_2020.json` explicitly defines the 2020 reproduction, uses LSODA with a 300-second maximum/output step, and ends after 9,676,800 seconds (112 days).

## Why the 513 MB archive is unnecessary

The official formatter reads only four files from `Simulation data/CSV output`: HPS/LED climate-model output and HPS/LED energy-use output. ZIP-directory inspection shows that all four are at the start of the archive and total about 22.04 MiB compressed. A byte-range request can retrieve them without downloading the remaining MATLAB source and output bundle.

The full 538,319,416-byte archive will not be downloaded. The smaller range is also still pending explicit approval, following the user's download rule and an automatic approval rejection.

## Public processed-data anomaly

The raw HPS and LED files each contain 32,257 uninterrupted five-minute records from 2009-10-19 15:15 through 2010-02-08 15:15, matching the methodology and the model's 112-day horizon.

In contrast, `dataHPS.csv` and `dataLED.csv` each contain 36,394 rows. Their time column continues for 4,137 records beyond the declared experiment, ending at 2010-02-23 00:00. Those trailing rows must never enter a score or fit. The frozen rule is to retain exactly the first 32,257 records through 2010-02-08 15:15.

## Safe execution order

After approval for the 22.04 MiB subset:

1. extract and CRC-check only the four required CSV files;
2. run the official v2.0.5 formatter and hash its generated inputs;
3. create an isolated environment with the version-pinned GreenLight dependencies;
4. run a one-day HPS climate smoke test using one worker and one numerical-library thread;
5. report runtime and load, then freeze exact metrics from the actual output schema;
6. run the four 112-day official simulations only after the smoke test passes.

This order prevents another sustained high-load run before the data boundary, software version and scoring contract are inspectable.

Sources: [official GreenLight repository](https://github.com/davkat1/GreenLight), [original evaluation dataset](https://doi.org/10.4121/78968e1b-eaea-4f37-89f9-2b98ba3ed865).
