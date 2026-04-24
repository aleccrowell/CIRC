"""Shared I/O utilities for reading and writing expression data."""
from pathlib import Path

import pandas as pd


def read_expression(source, data_type='r'):
    """Read an expression DataFrame from a file path, or return one unchanged.

    Parquet files (detected by ``.parquet`` extension) carry their own index
    and are returned as-is.  Tab-separated text files are read with the
    appropriate index column(s) for the given *data_type*.

    Parameters
    ----------
    source : str | Path | pd.DataFrame
        File path or an already-loaded DataFrame.  DataFrames are returned as
        a shallow copy with no further processing.
    data_type : str
        ``'r'`` — RNAseq-style; first column is the ``#`` gene ID index.
        ``'p'`` — proteomics-style; ``Peptide`` and ``Protein`` columns form a
        multi-index.  Only consulted for tab-separated text files.
    """
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = str(source)
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    if data_type == 'p':
        return pd.read_csv(path, sep='\t').set_index(['Peptide', 'Protein'])
    return pd.read_csv(path, sep='\t', header=0, index_col=0)


def write_expression(df, path):
    """Write an expression DataFrame to disk.

    Writes Apache Parquet when *path* ends with ``.parquet``; tab-separated
    text otherwise.

    Parameters
    ----------
    df : pd.DataFrame
    path : str | Path
    """
    path = str(path)
    if path.endswith('.parquet'):
        df.to_parquet(path)
    else:
        df.to_csv(path, sep='\t')


def sidecar_path(main_path, suffix):
    """Derive a sidecar file path that inherits the extension of the main output.

    Parameters
    ----------
    main_path : str | Path
        Path to the main output file, e.g. ``/data/out.parquet``.
    suffix : str
        String to insert before the extension, e.g. ``'_trends'``.

    Returns
    -------
    str
        Path with *suffix* prepended to the extension, e.g.
        ``/data/out_trends.parquet``.
    """
    p = Path(main_path)
    return str(p.with_name(p.stem + suffix + p.suffix))
