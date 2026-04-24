import argparse
import os
import re
import tempfile
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from circ.simulations import simulate  # noqa: F401  re-exported for backward compat


class analyze:
    def __init__(self, filename_classes):
        self.true_classes = pd.read_csv(filename_classes, sep='\t')
        self.tags = {}
        self.merged = []
        self.i = 0

    def run_bootjtk(self, filename, tag, size=50, workers=1):
        """
        Run bootjtk significance testing on a LIMBR-processed (or baseline) file.

        Parameters
        ----------
        filename : str
            Path to a LIMBR-processed, old_fashioned, or baseline output file.
        tag : str
            Label used to identify this dataset in ROC curves.
        size : int
            Number of bootstrap resamples per gene (default 50).
        workers : int
            Parallel worker processes for BooteJTK (default 1).
        """
        from circ.bootjtk import BooteJTK, CalcP
        import circ.bootjtk as _bootjtk_pkg

        ref_dir = os.path.join(os.path.dirname(_bootjtk_pkg.__file__), 'ref_files')

        df = pd.read_csv(filename, sep='\t', index_col=0)

        data_cols = [c for c in df.columns if re.match(r'^(ZT|CT)?\d', str(c))]
        df = df[data_cols]

        df.columns = [re.sub(r'_\d+$', '', str(c)) for c in df.columns]
        df.index.name = 'ID'
        df = df.replace('NULL', 'NA')

        reps = int(np.median(list(Counter(df.columns).values())))

        def _make_args(fn):
            return argparse.Namespace(
                filename=fn,
                means='DEFAULT', sds='DEFAULT', ns='DEFAULT',
                output='DEFAULT', pickle='DEFAULT',
                id_list='DEFAULT', null_list='DEFAULT',
                write=False, prefix='', reps=reps, size=size,
                workers=workers, waveform='cosine',
                width=os.path.join(ref_dir, 'asymmetries_02-22_by2.txt'),
                phase=os.path.join(ref_dir, 'phases_00-22_by2.txt'),
                period=os.path.join(ref_dir, 'period24.txt'),
                harding=False, normal=False,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, 'data.txt')
            df.to_csv(data_path, sep='\t')

            null_df = pd.DataFrame(
                np.random.normal(0, 1, df.shape),
                index=df.index,
                columns=df.columns,
            )
            null_df.index.name = 'ID'
            null_path = os.path.join(tmpdir, 'null.txt')
            null_df.to_csv(null_path, sep='\t')

            data_out, _, _ = BooteJTK.main(_make_args(data_path))
            null_out, _, _ = BooteJTK.main(_make_args(null_path))

            CalcP.main(argparse.Namespace(
                filename=data_out,
                null=null_out,
                fit=False,
            ))
            calcp_out = data_out.replace('.txt', '_GammaP.txt')

            self.add_data(calcp_out, tag)

    def add_data(self, filename_ejtk, tag, include_missing=True):
        self.tags[tag] = self.i
        ejtk = pd.read_csv(filename_ejtk, sep='\t')
        if include_missing:
            self.merged.append(pd.merge(
                self.true_classes[['Protein', 'Circadian']],
                ejtk[['ID', 'GammaBH']],
                left_on='Protein', right_on='ID', how='left',
            ))
        else:
            self.merged.append(pd.merge(
                self.true_classes[['Protein', 'Circadian']],
                ejtk[['ID', 'GammaBH']],
                left_on='Protein', right_on='ID', how='inner',
            ))
        self.merged[self.i].set_index('Protein', inplace=True)
        self.merged[self.i].drop('ID', axis=1, inplace=True)
        self.merged[self.i].fillna(1.0, inplace=True)
        self.i += 1

    def generate_roc_curve(self):
        for j in self.tags.keys():
            fpr, tpr, _ = roc_curve(
                self.merged[self.tags[j]]['Circadian'].values,
                1 - self.merged[self.tags[j]]['GammaBH'].values,
                pos_label=1,
            )
            plt.plot(fpr, tpr, label=j)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Comparison')
        plt.legend(loc="lower right")
        plt.savefig('ROC.pdf')

    def calculate_auc(self):
        out = {}
        for j in self.tags.keys():
            fpr, tpr, _ = roc_curve(
                self.merged[self.tags[j]]['Circadian'].values,
                1 - self.merged[self.tags[j]]['GammaBH'].values,
                pos_label=1,
            )
            out[j] = auc(fpr, tpr)
        self.roc_auc = out
