# SlowLab

**A benchmark for experimental-design agents when experiments are slow, noisy, expensive
and irreversible.**

Benchmarks for automated experimental design make experiments instantaneous, free and
infinitely repeatable. Real experiments are none of these. SlowLab is an executable
environment in which four conditions hold *at once*:

| Condition | How it appears here |
|---|---|
| **Slow** | a trial is a growing cycle. A task gives an agent two or three of them, for 420–630 simulated days in total. |
| **Noisy** | plant-to-plant, loop-to-loop, chamber-to-chamber and batch-to-batch variance, none of it removable by asking again. |
| **Costly** | a trial is priced in the objective's own units — running a bad treatment *is* the loss. |
| **Irreversible** | a submitted design is committed; a heat-damaged unit is out for the rounds it takes to recover. |

An agent runs a greenhouse. It designs an experiment, commits it, waits out the cycle,
sees the outcome, and repeats. Ground truth is the reduced state-variable TOMGRO crop
model (Jones, Kenig & Vallejos 1999) with a European greenhouse cost model, so the
difficulty is a property of the domain rather than a set of dials we tuned, and everything
— the objective, the price of experimenting, and both regrets — is denominated in net
margin per square metre per day.

The environment is **frozen at v1.0.0**. Its known defects are published rather than all
fixed, in [`ENVIRONMENT_v1.0.md`](ENVIRONMENT_v1.0.md); a benchmark's value is
comparability across agents, not correctness of the simulator.

---

## Install and run

```bash
pip install -r requirements.txt
python3 -m pytest -q                       # 92 passed
python3 scripts/run_baselines.py           # the four scripted reference strategies
```

An episode, in full:

```python
from slowlab import SlowLabEnv, TASKS, Design

env = SlowLabEnv(TASKS["Optimise"], seed=0)   # a seed draws a *site*, not a noise draw

while env.rounds_left:
    u = env.available_units()                 # excludes units still in heat recovery
    d = Design(
        treatments={"A": {"day_temp": 0.3, "density": 0.5},   # factors are coded to [0, 1]
                    "B": {"day_temp": 0.7, "density": 0.5}},
        allocation={"A": u[:6], "B": u[6:12]},
        randomization_seed=1,
    )
    r = env.submit_design(d)                  # a falsy Rejection, or the round index
    obs = env.advance()                       # one growing cycle passes
    env.interim_recommendation(x_now)         # scored every round, costs nothing

res = env.submit_recommendation(x_final)
print(res.simple_regret, res.cumulative_regret)
```

Factor values are coded to $[0, 1]$ over the task's range — `day_temp` spans 18–32 °C on
**Optimise**, `density` 2.2–4.5 plants m⁻². `env.validate_design(d)` answers the same
question as `submit_design` without committing anything.

The interface guarantees timing and feasibility and nothing else. It never reveals the
site parameters $\theta$, the instance distribution $P_\Theta$, or the forward model.
Feasibility checking is free and unlimited, and a rejected design costs no budget, so an
agent can never lose a cycle to a formatting mistake.

To run language models, see [`RUNNING.md`](RUNNING.md).

---

## The four tasks

Each is a stage an industrial campaign goes through, and each isolates a capability the
previous one does not need. The variance components, cycle length, per-unit cost and
commitment structure are identical across all four; only the question and the geometry
change.

| Task | What the agent must do | Factors | Facility | rounds × units | days |
|---|---|---|---|---|---|
| **Sanity** | close the loop at all | 1 (day temperature) | 4 × 4 | 2 × 8 | 420 |
| **Screen** | find which factors matter | 6 (management) | 8 × 3 | 2 × 12 | 420 |
| **Optimise** | find the optimum, respecting the facility | 2 (mixed level) | 4 × 3 | 3 × 12 | 630 |
| **Transfer** | survive an energy price shock | 4 (resource) | 8 × 3 | 3 × 24 | 630 |

*Facility* is chambers × loops. *Mixed level* means one chamber-level and one loop-level
factor, so a valid design has to be a split-plot. In **Transfer**, electricity and gas
both rise by a factor of 2.2 after the campaign ends and the agent must recommend again
**with no further experiments** — an agent that modelled revenue and the two cost streams
separately can re-solve; one that modelled only the margin cannot.

---

## What is measured

Four quantities per episode, never combined into one score.

**Realised regret** $\bar{\mathcal{R}}$ — the loss of the recommendation against the
site's true optimum, elicited after *every* round rather than only at the end.

**Cumulative regret** $\mathcal{R}_{\mathrm{cum}}$ — what the experiments themselves
forwent, in the same currency, summed over occupied slot-days. An agent that finds the
optimum by first running every bad treatment has answered the question and exhausted the
facility that paid for it.

The other two come from $\mathcal{R}^\star(D)$, the regret a *perfect reasoner* would
still carry given design $D$ — the prior and preposterior Bayes risk of the decision
problem, computed from an instance distribution no agent can see:

**Design efficiency** $c = [\mathcal{R}^\star(\varnothing) - \mathcal{R}^\star(D)] /
\mathcal{R}^\star(\varnothing)$ — the fraction of the available risk the experiments
removed. 0% for a design that teaches nothing, 100% for one that identifies the optimum.

**Decision efficiency** $\eta = \mathcal{R}^\star(D) / \bar{\mathcal{R}}$ — what the
design made available, over what the agent got from it. Near 100% when the recommendation
is as good as the data allowed.

The pair separates a campaign that *could not* have answered the question from one that
could and did not. Both are reported as ranks: the *level* of $\mathcal{R}^\star$ holds
only 0.81–0.92 between M = 80 and M = 320 atoms, while the rank correlation is 0.96–0.97.

---

## Headline results

Five models — DeepSeek-v4-flash, GPT-5.6-luna, Qwen3.8-27B, MiMo-v2.5, GLM-5.3-flash —
on all four tasks, twice each (bare and with two standard tools): 40 cells, 800 episodes,
20 instances per cell.

- **The benchmark is far from saturated.** No arm reaches
  $\mathcal{R}^\star(\varnothing)$, the ceiling a perfect reasoner carries having run
  nothing at all. Bare arms sit 1.8× to 23.4× above it, widest where the design problem
  is hardest.
- **The designs could not have answered the question.** Unaided, 99% of designs on Screen,
  92% on Transfer and 54% on Optimise are rank-deficient over the factors the task varies.
  The median design uses **two** distinct treatments replicated across the facility,
  whether the task needs two or seven. These are not careless designs — they declare a
  randomisation seed, replicate heavily and respect the control hierarchy. They buy
  precision with the units coverage needed.
- **Decisions then use a small fraction of what those designs allowed.** $\eta$ runs from
  4% to 46%, splitting by task rather than by model.
- **The two failures are independent.** Within task, $c$ and $\eta$ rank-correlate at
  +0.80 on Optimise and −0.80 on Transfer. A single score would have reported only their
  product.
- **Standard tools improve the design and damage the answer.** Adding a screening-design
  generator and a GP posterior-maximisation routine cuts rank-deficiency from 99% to 59%
  on Screen — and makes 16 of 20 pairs worse on realised regret. The inference tool
  inherits whatever the design supports: the same routine carries 0.0172 on its own
  reference campaign and 0.0711 fitted to a model's design on Sanity. Models defer to it
  in proportion to how wrong it is, and in 55% of turns the recommendation they state is
  the tool's proposed point to six digits.
- **The first cycle of data often makes the recommendation worse.** In 44% of episodes the
  recommendation after round 1 is worse than the one the same model made with no data at
  all.

Full tables, with standard errors and the paired tests, are in the paper.

---

## Repository layout

```
slowlab/        the environment (frozen at v1.0.0)
  env.py          episode loop, commitment, timing, feasibility
  tasks.py        the four tasks
  world.py        P_Theta: a seed draws a site (climate, prices, crop parameters)
  tomgro.py       reduced state-variable TOMGRO
  economics.py    greenhouse gross-margin model
  facility.py     chambers and loops; the control hierarchy
  design.py       the submitted design and rejection semantics
  achievable.py   R*(D): the regret a design supports
  eig.py          the estimator behind it
  agents.py       four scripted reference strategies
  llm.py          the language-model harness (prompt, retry loop, tools)
  providers.py    chat APIs over urllib; no SDK required

scripts/        every number in the paper has a script here
tests/          92 tests, including test_frozen.py, which fails if ground truth moves
paper/          the LaTeX source
results/        summary JSON per model and task
```

**`scripts/` is the reproduction path.** Nothing in the paper is a one-off analysis:

| Script | What it produces |
|---|---|
| `run_llm.py`, `run_all_llm.sh` | the language-model episodes |
| `run_baselines.py` | the four scripted reference strategies |
| `run_achievable.py`, `rescore_cv.py` | $\mathcal{R}^\star(D)$ under the cross-validated estimator |
| `make_tasks_table.py`, `make_llm_tables.py`, `make_reference_table.py`, `make_diag_table.py` | Tables 2–5 |
| `verify_results_claims.py` | re-derives every number in the results section, printing `paper: X / now: Y` |
| `tool_regret.py`, `tool_design_width.py`, `tool_hyper_control.py` | the three legs of the tool analysis |
| `factor_value.py` | the cost of setting a factor wrongly, against the value of locating it |
| `rstar_convergence.py`, `eig_rank_stability.py` | why the estimator is read as ranks |
| `check_crossrefs.py` | orphan labels, dangling refs, duplicates |

Transcripts for all 800 episodes and the atom-set pickles are published as a GitHub
Release asset rather than committed here.

---

## Adding a domain

Several domains satisfy the four conditions: microbial evolution, bioprocess scale-up,
materials ageing, dose-finding trials. Adding one requires no environment code — a domain
supplies a forward model $f_\theta$, an instance distribution $P_\Theta$, a facility with
its control hierarchy, and a price list. The appendix of the paper sets out what a new
domain has to bring.

---

## Citation

See [`CITATION.cff`](CITATION.cff). Licensed under the terms in [`LICENSE`](LICENSE).
