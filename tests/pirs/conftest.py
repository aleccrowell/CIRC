import pytest
import numpy as np
import pandas as pd


def make_ranker_tsv(tmp_path, n_timepoints=4, n_reps=3, n_genes=8, prefix="ZT"):
    """Write a tab-separated file in the format expected by ranker/rsd_ranker."""
    tpoints = [(i + 1) * 2 for i in range(n_timepoints)]  # 2, 4, 6, 8, ...
    cols = [f"{prefix}{t:02d}_{r}" for t in tpoints for r in range(1, n_reps + 1)]

    np.random.seed(0)
    rows = {}
    for i in range(n_genes):
        rows[f"gene_{i}"] = np.random.normal(1.0, 0.05, len(cols))

    # One clearly rhythmic gene — values ramp hard across timepoints so ANOVA removes it
    rhythmic = [float(t_idx * 20) for t_idx in range(n_timepoints) for _ in range(n_reps)]
    rows["rhythmic_gene"] = rhythmic

    df = pd.DataFrame(rows, index=cols).T
    df.index.name = "#"
    path = tmp_path / "test_data.txt"
    df.to_csv(path, sep="\t")
    return str(path)


@pytest.fixture
def tsv_file(tmp_path):
    return make_ranker_tsv(tmp_path)


@pytest.fixture
def tsv_no_ct(tmp_path):
    return make_ranker_tsv(tmp_path, prefix="")
