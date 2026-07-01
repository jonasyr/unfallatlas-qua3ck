import numpy as np
import pandas as pd

from unfallatlas.features.temporal import cyclic_encode


def test_cyclic_encode_adds_sin_cos_columns():
    df = pd.DataFrame({"USTUNDE": [0, 6, 12, 18, 23]})
    out = cyclic_encode(df, "USTUNDE", period=24)
    assert {"USTUNDE_sin", "USTUNDE_cos"}.issubset(out.columns)
    assert len(out) == 5


def test_cyclic_encode_does_not_mutate_input():
    df = pd.DataFrame({"USTUNDE": [0, 12]})
    cyclic_encode(df, "USTUNDE", period=24)
    assert list(df.columns) == ["USTUNDE"]


def test_cyclic_encode_wraps_hour_0_and_24_to_the_same_point():
    df = pd.DataFrame({"USTUNDE": [0, 24]})
    out = cyclic_encode(df, "USTUNDE", period=24)
    np.testing.assert_allclose(out["USTUNDE_sin"].iloc[0], out["USTUNDE_sin"].iloc[1], atol=1e-9)
    np.testing.assert_allclose(out["USTUNDE_cos"].iloc[0], out["USTUNDE_cos"].iloc[1], atol=1e-9)


def test_cyclic_encode_hour_23_is_closer_to_0_than_to_12():
    near = cyclic_encode(pd.DataFrame({"USTUNDE": [23, 0]}), "USTUNDE", period=24)
    far = cyclic_encode(pd.DataFrame({"USTUNDE": [23, 12]}), "USTUNDE", period=24)
    near_dist = np.hypot(
        near["USTUNDE_sin"].iloc[0] - near["USTUNDE_sin"].iloc[1],
        near["USTUNDE_cos"].iloc[0] - near["USTUNDE_cos"].iloc[1],
    )
    far_dist = np.hypot(
        far["USTUNDE_sin"].iloc[0] - far["USTUNDE_sin"].iloc[1],
        far["USTUNDE_cos"].iloc[0] - far["USTUNDE_cos"].iloc[1],
    )
    assert near_dist < far_dist
