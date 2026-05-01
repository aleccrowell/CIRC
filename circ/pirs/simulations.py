from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from circ.simulations import simulate  # noqa: F401  re-exported for backward compat


class analyze:
    def __init__(self) -> None:
        self.true_classes = pd.DataFrame()
        self.merged = pd.DataFrame()

    def add_classes(self, filename_classes: str, rep: int = 0) -> None:
        tc = pd.read_csv(filename_classes, sep="\t")
        tc["rep"] = rep
        self.true_classes = pd.concat([self.true_classes, tc])

    def add_data(self, filename_pirs: str, tag: str, rep: int = 0) -> None:
        ranks = pd.read_csv(filename_pirs, sep="\t")
        ranks["method"] = tag
        ranks["rep"] = rep
        ranks["score"] = ranks["score"].fillna(ranks["score"].max())
        if self.merged.empty:
            self.merged = ranks
        else:
            self.merged = pd.concat([self.merged, ranks])

    def _build_curves(self) -> pd.DataFrame:
        """Compute per-method precision-recall curves and return as a DataFrame."""
        curves = pd.DataFrame(columns=["precision", "recall", "method", "rep"])
        melted = (
            self.merged.pivot_table(
                index=["rep", "method"], columns="#", values="score"
            )
            .fillna(self.merged.score.max())
            .reset_index()
            .melt(id_vars=["rep", "method"], value_name="score")
        )
        for rep in melted.rep.unique():
            for method in melted.method.unique():
                pr = pd.merge(
                    self.true_classes[self.true_classes["rep"] == rep],
                    melted[(melted.rep == rep) & (melted.method == method)],
                    on=["#", "rep"],
                )
                precision, recall, _ = precision_recall_curve(
                    pr["Const"].values, 1 / pr["score"].values, pos_label=1
                )
                temp = pd.DataFrame(
                    {
                        "precision": precision,
                        "recall": recall,
                        "method": method,
                        "rep": rep,
                    }
                )
                curves = pd.concat([curves, temp], sort=False)
        return curves

    def generate_pr_curve(self, outpath: str = "PR.pdf") -> Any:
        """Build precision-recall curves and save to *outpath* (default PR.pdf)."""
        from circ.visualization.benchmarks import pr_curve

        self.curves = self._build_curves()
        baseline = float(np.mean(self.true_classes["Const"]))
        ax = pr_curve(self.curves, baseline=baseline, outpath=outpath)
        return ax
