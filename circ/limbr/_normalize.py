"""Shared normalization helpers used by both batch_fx.sva and old_fashioned."""

import pandas as pd


def _pool_norm(df: pd.DataFrame, dmap: dict) -> pd.DataFrame:
    """Divide each sample column by its matched pooled-control column.

    Parameters
    ----------
    df   : DataFrame with sample and pool columns.
    dmap : dict mapping each sample column to its pool number (integer).
    """
    newdf = pd.DataFrame(index=df.index)
    for column in df.columns.values:
        if "pool" not in column:
            newdf[column] = df[column].div(
                df["pool_" + "%02d" % dmap[column]], axis="index"
            )
    nonpool = [i for i in newdf.columns if "pool" not in i]
    return newdf[nonpool]


def _qnorm(df: pd.DataFrame) -> pd.DataFrame:
    """Quantile-normalize df column-wise to a shared rank-mean reference."""
    ref = (
        pd.concat(
            [df[col].sort_values().reset_index(drop=True) for col in df],
            axis=1,
            ignore_index=True,
        )
        .mean(axis=1)
        .values
    )
    for i in range(len(df.columns)):
        df = df.sort_values(df.columns[i])
        df[df.columns[i]] = ref
    return df.sort_index()
