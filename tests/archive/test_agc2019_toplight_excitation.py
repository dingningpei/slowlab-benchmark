import pytest

from scripts.agc.audit_agc2019_toplight_excitation import summarize
from slowlab.archive.agc_lighting import LED_FIELDS


def sample(q, values):
    return q, dict(zip(LED_FIELDS, values))


def test_excitation_reports_rank_and_feasible_energy_partition():
    rows = [sample(10, (1, 0, 0, 0)), sample(20, (0, 1, 0, 0)),
            sample(30, (0, 0, 1, 0)), sample(40, (0, 0, 0, 1))]
    result = summarize(rows, zeta=5.4, eta_nir=0.02, eta_cool=0)
    assert result["channel_matrix_rank"] == 4
    assert result["unique_channel_vectors"] == 4
    assert result["seed_energy_partition_feasible"]


def test_infeasible_radiative_partition_is_detected():
    result = summarize([sample(1, (0, 1, 0, 0))], zeta=5.4,
                       eta_nir=0.9, eta_cool=0)
    assert not result["seed_energy_partition_feasible"]


@pytest.mark.parametrize("args", [(0, .1, 0), (5, -.1, 0), (5, .7, .4)])
def test_invalid_partition_parameters_fail(args):
    with pytest.raises(ValueError):
        summarize([sample(10, (1, 0, 0, 0))], *args)
