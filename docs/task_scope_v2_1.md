# Version 2.1 task scope

Version 2.1 reports three core benchmark tasks on the unchanged environment v2.0.0.

| Status | Task | Role in the benchmark |
| --- | --- | --- |
| Core | Sanity | Closed-loop interface and one-factor evaluator calibration |
| Core | Optimise | Primary split-plot experimental-design task |
| Core | Transfer | Recommendation after a held-out energy-price shift |
| Experimental | Screen | Archived configuration for future screening-task development |

`Screen` remains executable so archived results and development experiments can be
reproduced. It is excluded from Version 2.1 model comparisons because its current output is
an operating recommendation and its endpoint is final regret. Those choices evaluate scarce
high-dimensional optimisation, not whether the agent identified which factors matter.

A future screening benchmark must ask the agent to return:

- a ranking or top-k set of factors;
- an effect direction for each selected factor;
- confidence or uncertainty for each judgement.

It must then score ranking quality, sign accuracy, confidence calibration, and top-k
selection directly. Final operating regret may remain a secondary outcome, but it cannot be
the evidence for a screening claim.
