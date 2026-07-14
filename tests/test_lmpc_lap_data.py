import numpy as np
import pytest

from controllers.lmpc.lap_data import LapSample, build_lap_arrays


def make_samples() -> list[LapSample]:
    return [
        LapSample(np.array([10.0, 1.0, 2.0, 3.0, 4.0, 5.0]), 0.10, 1.0),
        LapSample(np.array([20.0, 6.0, 7.0, 8.0, 9.0, 10.0]), 0.20, 1.5),
        LapSample(np.array([30.0, 11.0, 12.0, 13.0, 14.0, 15.0]), 0.30, 1.25),
    ]


def test_build_lap_arrays_aligns_states_and_realized_inputs() -> None:
    samples = make_samples()

    x_lap, u_lap, J_lap = build_lap_arrays(samples, 0.25)

    assert x_lap.shape == (6, 3)
    assert u_lap.shape == (2, 2)
    np.testing.assert_array_equal(J_lap, [2.0, 1.0, 0.0])
    np.testing.assert_array_equal(x_lap[:, -1], samples[-1].x)
    np.testing.assert_allclose(u_lap[0], [2.0, -1.0])
    np.testing.assert_allclose(u_lap[1], [0.10, 0.20])
    assert 0.30 not in u_lap[1]
    assert not np.array_equal(u_lap[0], np.diff(x_lap[0]) / 0.25)
    assert x_lap.dtype == u_lap.dtype == J_lap.dtype == np.float64
    assert x_lap.flags.c_contiguous
    assert u_lap.flags.c_contiguous
    assert J_lap.flags.c_contiguous


@pytest.mark.parametrize("dt", [0.0, -0.25, np.nan, np.inf])
def test_build_lap_arrays_rejects_invalid_dt(dt: float) -> None:
    with pytest.raises(ValueError, match="dt must be finite and strictly positive"):
        build_lap_arrays(make_samples(), dt)


def test_build_lap_arrays_requires_two_samples() -> None:
    with pytest.raises(ValueError, match="at least two"):
        build_lap_arrays(make_samples()[:1], 0.25)


def test_build_lap_arrays_rejects_wrong_state_shape() -> None:
    samples = make_samples()
    samples[1] = LapSample(np.zeros(5), 0.20, 1.5)
    with pytest.raises(ValueError, match=r"shape \(6,\)"):
        build_lap_arrays(samples, 0.25)


@pytest.mark.parametrize("field", ["state", "speed", "steering"])
def test_build_lap_arrays_rejects_nonfinite_samples(field: str) -> None:
    samples = make_samples()
    if field == "state":
        state = samples[1].x.copy()
        state[2] = np.nan
        samples[1] = LapSample(state, samples[1].actual_delta, samples[1].raw_speed)
    elif field == "speed":
        samples[1] = LapSample(samples[1].x, samples[1].actual_delta, np.nan)
    else:
        samples[1] = LapSample(samples[1].x, np.nan, samples[1].raw_speed)

    with pytest.raises(ValueError, match="finite"):
        build_lap_arrays(samples, 0.25)
