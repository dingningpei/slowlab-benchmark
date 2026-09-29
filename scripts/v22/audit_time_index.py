"""Reproduce the pinned upstream index expression; not a GreenLight runtime test."""
import json
import numpy as np


def main():
    times = np.array([0.0, 300.0, 600.0])
    query = np.array([0.0, 150.0, 300.0, 450.0])
    upstream = np.clip(times.searchsorted(query), 0, len(times) - 1)
    causal = times.searchsorted(query, side="right") - 1
    future = times[upstream] > query
    assert future.tolist() == [False, True, False, True]
    assert np.all(times[causal] <= query)
    assert times.searchsorted(-1.0, side="right") - 1 == -1  # no available record
    print(json.dumps({
        "scope": "index-expression reproduction only; no simulator executed",
        "query_s": query.tolist(),
        "upstream_selected_s": times[upstream].tolist(),
        "last_available_s": times[causal].tolist(),
        "future_selection_confirmed": bool(np.any(future)),
    }, indent=2))


if __name__ == "__main__":
    main()
