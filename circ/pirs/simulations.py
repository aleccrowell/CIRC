import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve

from circ.simulations import simulate  # noqa: F401  re-exported for backward compat


class analyze:
    def __init__(self):
        self.true_classes = pd.DataFrame()
        self.merged = pd.DataFrame()

    def add_classes(self, filename_classes, rep=0):
        tc = pd.read_csv(filename_classes, sep='\t')
        tc['rep'] = rep
        self.true_classes = pd.concat([self.true_classes, tc])

    def add_data(self, filename_pirs, tag, rep=0):
        ranks = pd.read_csv(filename_pirs, sep='\t')
        ranks['method'] = tag
        ranks['rep'] = rep
        ranks['score'] = ranks['score'].fillna(ranks['score'].max())
        if self.merged.empty:
            self.merged = ranks
        else:
            self.merged = pd.concat([self.merged, ranks])

    def generate_pr_curve(self):
        self.curves = pd.DataFrame(columns=['precision', 'recall', 'method'])
        colors = ["windows blue", "amber", "light grey", "black"]
        self.merged = (
            self.merged
            .pivot_table(index=['rep', 'method'], columns='#', values='score')
            .fillna(self.merged.score.max())
            .reset_index()
            .melt(id_vars=['rep', 'method'], value_name='score')
        )
        for rep in self.merged.rep.unique():
            for method in self.merged.method.unique():
                pr = pd.merge(
                    self.true_classes[self.true_classes['rep'] == rep],
                    self.merged[
                        (self.merged.rep == rep) & (self.merged.method == method)
                    ],
                    on=['#', 'rep'],
                )
                precision, recall, _ = precision_recall_curve(
                    pr['Const'].values, 1 / pr['score'].values, pos_label=1
                )
                temp = pd.DataFrame({
                    'precision': precision,
                    'recall': recall,
                    'method': method,
                    'rep': rep,
                })
                self.curves = pd.concat([self.curves, temp], sort=False)
        ax = sns.lineplot(
            x='recall', y='precision', hue='method', units='rep',
            palette=sns.xkcd_palette(colors), estimator=None, data=self.curves,
        )
        ax.set_aspect(aspect=0.5)
        plt.plot(
            [0, 1],
            [np.mean(self.true_classes['Const']), np.mean(self.true_classes['Const'])],
            color='r', linestyle=':',
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.setp(ax.lines, linewidth=0.5)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall Comparison')
        leg = plt.legend(loc='upper center', bbox_to_anchor=(1.2, 0.8), shadow=True)
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        plt.savefig('PR.pdf', dpi=25)
        plt.close()
