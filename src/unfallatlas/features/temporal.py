"""Cyclic encoding for periodic time features (U-phase §10 contract).

Only the three columns the U-phase decided on are ever encoded here:
UMONAT (period 12), USTUNDE (period 24), UWOCHENTAG (period 7). This module
does not add holiday flags, weekend flags, or any other temporal feature —
those were not part of the U-phase §10 handover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cyclic_encode(df: pd.DataFrame, column: str, period: int) -> pd.DataFrame:
    """Return a copy of ``df`` with ``{column}_sin`` / ``{column}_cos`` added.

    The circular encoding ensures the model sees hour 23 and hour 0 as
    adjacent rather than 23 apart — the reason U-phase §10 mandates this
    over a raw integer or one-hot encoding for UMONAT/USTUNDE/UWOCHENTAG.
    """
    out = df.copy()
    radians = 2 * np.pi * out[column].astype(float) / period
    out[f"{column}_sin"] = np.sin(radians)
    out[f"{column}_cos"] = np.cos(radians)
    return out
