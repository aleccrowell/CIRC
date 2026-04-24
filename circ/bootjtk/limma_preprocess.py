"""Data I/O and reshaping for the limma preprocessing pipeline.

Handles reading TSV files, parsing ZT/CT time-point labels, deduplicating
row/column names, and writing pivoted output files — the data plumbing that
was previously embedded in the R scripts.
"""
import re
import pandas as pd
import numpy as np


def read_timeseries(fn):
    """Read a tab-separated timeseries file.

    Handles both '#' and 'ID' as the first-column header.

    Returns:
        df       -- DataFrame with row IDs as index, raw label strings as columns
        raw_cols -- list of raw column-header strings (e.g. ['ZT0', 'ZT2', ...])
    """
    with open(fn) as f:
        raw_cols = f.readline().rstrip('\n').split('\t')[1:]
    try:
        df = pd.read_table(fn, index_col='ID')
    except (ValueError, KeyError):
        df = pd.read_table(fn, index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df, raw_cols


def parse_timepoint_label(label):
    """Strip ZT, CT, or leading X prefix and return a float time value.

    >>> parse_timepoint_label('ZT4')
    4.0
    >>> parse_timepoint_label('CT12')
    12.0
    >>> parse_timepoint_label('X8')
    8.0
    >>> parse_timepoint_label('0')
    0.0
    """
    label = re.sub(r'^X', '', str(label).strip())
    label = re.sub(r'^ZT', '', label)
    label = re.sub(r'^CT', '', label)
    label = label.split('_')[0]
    return float(label)


def deduplicate_timepoints(times, period):
    """Disambiguate duplicate time values by adding period offsets.

    For each value that duplicates an earlier one in the list, adds multiples
    of *period* until the value is unique. Matches the algorithm used in the
    R vash script.

    Args:
        times  -- sequence of numeric time values (may contain duplicates)
        period -- period value (24.0 for circadian data)

    Returns:
        list of unique float time values with the same length as *times*
    """
    times = [float(t) for t in times]
    while len(set(times)) < len(times):
        seen: set = set()
        for i, t in enumerate(times):
            if t in seen:
                times[i] += period
            else:
                seen.add(t)
    return times


def deduplicate_rownames(rownames):
    """Append -xxx{n} suffixes to duplicate row names, matching R behavior.

    Args:
        rownames -- list of row-name strings (may contain duplicates)

    Returns:
        list of unique strings; duplicates get '-xxx1', '-xxx2', ... appended
    """
    rownames = list(rownames)
    counter = 1
    while len(set(rownames)) < len(rownames):
        seen: dict = {}
        new = []
        for name in rownames:
            if name in seen:
                new.append(f'{name}-xxx{counter}')
            else:
                seen[name] = True
                new.append(name)
        rownames = new
        counter += 1
    return rownames


def prepare_timeseries(fn, period):
    """Full preprocessing pipeline: read, parse time labels, deduplicate.

    Args:
        fn     -- path to input TSV file
        period -- circadian period (usually 24.0)

    Returns:
        df           -- DataFrame with numeric float column names and a
                        deduplicated 'ID' index
        unique_times -- sorted list of unique time points mod period
    """
    df, raw_cols = read_timeseries(fn)
    times = deduplicate_timepoints(
        [parse_timepoint_label(c) for c in raw_cols], period
    )
    df.columns = pd.Index(times, dtype=float)
    df.index = pd.Index(deduplicate_rownames(list(df.index)), name='ID')
    unique_times = sorted({t % period for t in times})
    return df, unique_times


def write_limma_outputs(long_df, prefix, suffix):
    """Write wide-format output files from long-format limma/vash results.

    Produces four tab-separated files readable by BooteJTK.read_in():
      {prefix}_Means_{suffix}.txt
      {prefix}_Sds_{suffix}.txt
      {prefix}_Ns_{suffix}.txt
      {prefix}_Sds-pre_{suffix}.txt

    Args:
        long_df -- DataFrame with columns ID, Time, Mean, SD, SDpre, N
        prefix  -- output path prefix (e.g. '/path/to/sample')
        suffix  -- output suffix (e.g. 'postLimma' or 'postVash')
    """
    for col, name in [('Mean', 'Means'), ('SD', 'Sds'), ('N', 'Ns'), ('SDpre', 'Sds-pre')]:
        wide = long_df.pivot(index='ID', columns='Time', values=col)
        wide.columns.name = None
        wide.to_csv(f'{prefix}_{name}_{suffix}.txt', sep='\t', na_rep='NA')
