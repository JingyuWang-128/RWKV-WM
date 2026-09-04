import numpy as np

from cape_wm.cem import CEMConfig, cem_optimize


def test_cem_finds_quadratic_minimum_reproducibly():
    config = CEMConfig(samples=512, elites=64, iterations=6, seed=7)

    def objective(population):
        return np.square(population - 0.37).sum(axis=1)

    first = cem_optimize(objective, (3,), -1.0, 1.0, config)
    second = cem_optimize(objective, (3,), -1.0, 1.0, config)
    np.testing.assert_allclose(first.value, second.value)
    np.testing.assert_allclose(first.value, np.full(3, 0.37), atol=0.05)


def test_cem_accepts_empirical_initial_population_outside_default_bounds():
    initial = np.asarray([[-5.0], [4.0], [6.0], [7.0]], dtype=np.float32)
    result = cem_optimize(
        lambda population: np.square(population[:, 0] - 4.0),
        (1,),
        -3.0,
        3.0,
        CEMConfig(samples=4, elites=2, iterations=1, seed=1, clip=False),
        initial_population=initial,
    )

    assert result.value.item() == 4.0
