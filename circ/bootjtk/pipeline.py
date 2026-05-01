#!/usr/bin/env python
"""
Created on Nov 1, 2015
@author: Alan L. Hutchison, alanlhutchison@uchicago.edu, Aaron R. Dinner Group, University of Chicago

This script is a bootstrapped expansion of the eJTK method described in

Hutchison AL, Maienschein-Cline M, and Chiang AH et al. Improved statistical methods enable greater sensitivity in rhythm detection for genome-wide data, PLoS Computational Biology 2015 11(3): e 1004094. doi:10.1371/journal.pcbi.1004094

This script bootstraps time series and provides phase and tau distributions from those bootstraps to allow for measurement of the variance on phase and tau estimates.


Please use ./BooteJTK -h to see the help screen for further instructions on running this script.

"""

VERSION = "1.0"

# import cmath

# import scipy.stats as ss
import numpy as np

# from scipy.stats import kendalltau as kt
import pandas as pd

# from operator import itemgetter
import argparse
import os.path

import hashlib
import json
import shutil
from pathlib import Path
import os

from . import BooteJTK
from . import CalcP
from .limma_preprocess import prepare_timeseries, write_limma_outputs
from .limma_voom import run_vooma_ebayes, run_vooma_vash

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_NULL_CACHE_DIR = Path.home() / ".cache" / "circ" / "null_cache"


def _null_cache_key(
    header,
    period_file,
    phase_file,
    width_file,
    waveform,
    reps,
    size,
    limma,
    vash,
    noreps,
):
    """Return a hex digest that uniquely identifies a null distribution configuration."""

    def _read(path):
        try:
            return open(path).read()
        except (OSError, TypeError):
            return str(path)

    payload = json.dumps(
        {
            "header": sorted(header),
            "period": _read(period_file),
            "phase": _read(phase_file),
            "width": _read(width_file),
            "waveform": str(waveform),
            "reps": int(reps),
            "size": int(size),
            "limma": bool(limma),
            "vash": bool(vash),
            "noreps": bool(noreps),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _cached_null_path(key):
    # Filename must contain 'boot' so CalcP reads the TauMean column correctly.
    return _NULL_CACHE_DIR / f"null_boot_{key}.txt"


def _load_null_cache(key):
    """Return path to cached null output if it exists, else None."""
    p = _cached_null_path(key)
    return str(p) if p.exists() else None


def _save_null_cache(key, src_path):
    _NULL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, _cached_null_path(key))


def main(args):

    fn = args.filename

    ### INTEGRATING THIS INTO THE R CODE
    # fn_means = args.means
    # fn_sds = args.sds
    # fn_ns = args.ns

    prefix = args.prefix
    fn_waveform = args.waveform
    fn_period = args.period
    fn_phase = args.phase
    fn_width = args.width
    fn_out = args.output
    fn_out_pkl = args.pickle  # This is the output file which could be read in early
    fn_list = args.id_list  # This is the the list of ids to go through
    fn_null_list = args.null_list  # These are geneIDs to be used to estimate the SD
    size = int(args.size)
    reps = int(args.reps)

    try:
        assert ".txt" in fn or ".txt" in args.means
    except AssertionError:
        print("Please make the suffix of your raw data file or means file .txt")
        assert 0

    if not args.basic:
        args.limma = True
        args.vash = True
    elif args.basic and args.limma and not args.vash:
        args.limma = True
        args.vash = False

    print("Basic:", args.basic)
    print("Limma:", args.limma)
    print("Vash:", args.vash)

    # fn_null = args.null
    if args.noreps:
        args.prefix = "NoRepSD_" + args.prefix
        print("No replicates, skipping Limma procedure")
        print("Estimating time point variance from arrhythmic genes")
        try:
            df = pd.read_table(fn, index_col="ID")
        except ValueError:
            df = pd.read_table(fn, index_col="#")
        except ValueError:
            print('Header needs to begin with "ID" or with "#"')

        j = pd.read_table(args.jtk, index_col="ID")
        mean = df.loc[j[j.GammaP > 0.8].index].std(axis=1).dropna().mean()

        df_sds = pd.DataFrame(
            np.ones(df.shape) * mean, index=df.index, columns=df.columns
        )
        df_ns = pd.DataFrame(np.ones(df.shape), index=df.index, columns=df.columns)
        fn_sds = fn.replace(".txt", "_Sds_noRepsEst.txt")
        df_sds.to_csv(fn_sds, na_rep=np.nan, sep="\t")
        fn_ns = fn.replace(".txt", "_Ns_noRepsEst.txt")
        df_sds.to_csv(fn_ns, na_rep=np.nan, sep="\t")

        args.means = fn
        args.sds = fn_sds
        args.ns = fn_ns

    elif args.limma:
        pref = fn.replace(".txt", "")
        period_val = float(pd.read_table(fn_period, header=None)[0][0])

        if not args.vash and fn is not None:
            print("Running the Limma commands")
            args.prefix = "Limma_" + args.prefix
            suffix = "postLimma"
        elif args.vash and fn is not None:
            print("Running the Vash commands")
            args.prefix = "Vash_" + args.prefix
            suffix = "postVash"

        args.means = fn.replace(".txt", f"_Means_{suffix}.txt")
        args.sds = fn.replace(".txt", f"_Sds_{suffix}.txt")
        args.ns = fn.replace(".txt", f"_Ns_{suffix}.txt")

        df_clean, _ = prepare_timeseries(fn, period_val)
        preprocessor = run_vooma_vash if args.vash else run_vooma_ebayes
        long_df = preprocessor(df_clean, period_val, rnaseq=args.rnaseq)
        write_limma_outputs(long_df, pref, suffix)
    else:
        print("Using neither Limma nor Vash")
        pass

    fn_out, fn_out_pkl, header = BooteJTK.main(args)

    # args.output = fn_out.replace('boot','NULL1000-boot')
    # args.pickle = fn_out_pkl.replace('boot','NULL1000-boot')

    # print args.pickle
    # print args.output
    null_key = _null_cache_key(
        header,
        fn_period,
        fn_phase,
        fn_width,
        fn_waveform,
        reps,
        size,
        args.limma,
        args.vash,
        args.noreps,
    )
    fn_null_out = _load_null_cache(null_key)

    if fn_null_out is not None:
        print(
            f"Null distribution cache hit (key={null_key[:8]}...) — skipping null BooteJTK run."
        )
    else:
        fn_null = fn.replace(".txt", "_NULL1000.txt")

        sims = 1000
        with open(fn_null, "w") as g:
            g.write("\t".join(["#"] + header) + "\n")
            for i in range(sims):
                line = ["wnoise_" + str(i)] + [
                    str(v) for v in np.random.normal(0, 1, len(header))
                ]
                g.write("\t".join(line) + "\n")

        args.filename = fn_null

        if args.noreps:
            print("No replicates, skipping Limma procedure")
            print("Estimating time point variance from arrhythmic genes")
            try:
                df = pd.read_table(fn, index_col="ID")
            except ValueError:
                df = pd.read_table(fn, index_col="#")
            except ValueError:
                print('Header needs to begin with "ID" or with "#"')

            j = pd.read_table(args.jtk, index_col="ID")
            mean = df.loc[j[j.GammaP > 0.8].index].std(axis=1).dropna().mean()

            df_sds = pd.DataFrame(
                np.ones(df.shape) * mean, index=df.index, columns=df.columns
            )
            df_ns = pd.DataFrame(np.ones(df.shape), index=df.index, columns=df.columns)
            fn_sds = fn_null.replace(".txt", "_Sds_noRepsEst.txt")
            df_sds.to_csv(fn_sds, na_rep=np.nan, sep="\t")
            fn_ns = fn_null.replace(".txt", "_Ns_noRepsEst.txt")
            df_sds.to_csv(fn_ns, na_rep=np.nan, sep="\t")

            args.means = fn_null
            args.sds = fn_sds
            args.ns = fn_ns
        elif args.limma:
            pref_null = fn_null.replace(".txt", "")
            if not args.vash:
                suffix_null = "postLimma"
                args.means = fn_null.replace(".txt", "_Means_postLimma.txt")
                args.sds = fn_null.replace(".txt", "_Sds_postLimma.txt")
                args.ns = fn_null.replace(".txt", "_Ns_postLimma.txt")
            else:
                suffix_null = "postVash"
                args.means = fn_null.replace(".txt", "_Means_postVash.txt")
                args.sds = fn_null.replace(".txt", "_Sds_postVash.txt")
                args.ns = fn_null.replace(".txt", "_Ns_postVash.txt")

            df_null_clean, _ = prepare_timeseries(fn_null, period_val)
            preprocessor = run_vooma_vash if args.vash else run_vooma_ebayes
            long_null = preprocessor(df_null_clean, period_val, rnaseq=args.rnaseq)
            write_limma_outputs(long_null, pref_null, suffix_null)
        else:
            pass

        fn_null_out, _, _ = BooteJTK.main(args)
        _save_null_cache(null_key, fn_null_out)
        print(f"Null distribution cached (key={null_key[:8]}...).")

    args.filename = fn_out
    args.null = fn_null_out
    args.fit = ""
    CalcP.main(args)


def __create_parser__():
    p = argparse.ArgumentParser(
        description="Bootstrap empirical JTK_CYCLE (eJTK) for circadian rhythm detection. "
        "See Hutchison et al. PLoS Comput Biol 2015 11(3):e1004094.",
        epilog="Please contact the corresponding author if you have any questions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", "-V", action="version", version="%(prog)s " + VERSION)

    # p.add_argument("-t", "--test",
    #               action='store_true',
    #               default=False,
    #               help="run the Python unittest testing suite")
    p.add_argument(
        "-o",
        "--output",
        dest="output",
        action="store",
        metavar="filename string",
        type=str,
        default="DEFAULT",
        help="You want to output something. If you leave this blank, _jtkout.txt will be appended to your filename",
    )

    p.add_argument(
        "-k",
        "--pickle",
        dest="pickle",
        metavar="filename string",
        type=str,
        action="store",
        default="DEFAULT",
        help='Should be a file with phases you wish to search for listed in a single column separated by newlines.\
                          Provided file is "period_24.txt"',
    )

    p.add_argument(
        "-l",
        "--list",
        dest="id_list",
        metavar="filename string",
        type=str,
        action="store",
        default="DEFAULT",
        help="A filename of the ids to be run in this time series. If time is running out on your job, this will be compared to the ids that have already been completed and a file will be created stating what ids remain to be analyzed.",
    )

    p.add_argument(
        "-n",
        "--null",
        dest="null_list",
        metavar="filename string",
        type=str,
        action="store",
        default="DEFAULT",
        help="A filename of the ids upon which to calculate the standard deviation. These ids are non-cycling, so the standard deviation can be taken across the entire time series. Using this argument is useless if the bootstraps have already been performed.",
    )

    analysis = p.add_argument_group(title="JTK_CYCLE analysis options")

    analysis.add_argument(
        "-f",
        "--filename",
        dest="filename",
        action="store",
        metavar="filename string",
        type=str,
        default="DEFAULT",
        help="This is the filename of the data series you wish to analyze.\
                   The data should be tab-spaced. The first row should contain a # sign followed by the time points with either CT or ZT preceding the time point (such as ZT0 or ZT4). Longer or shorter prefixes will not work. The following rows should contain the gene/series ID followed by the values for every time point. Where values are not available NA should be put in it's place.",
    )
    analysis.add_argument(
        "-F",
        "--means",
        dest="means",
        action="store",
        metavar="filename string",
        default="DEFAULT",
        type=str,
        help="This is the filename of the time point means of the data series you wish to analyze.\
                   The data should be tab-spaced. The first row should contain a # sign followed by the time points with either CT or ZT preceding the time point (such as ZT0 or ZT4). Longer or shorter prefixes will not work. The following rows should contain the gene/series ID followed by the values for every time point. Where values are not available NA should be put in it's place.",
    )

    analysis.add_argument(
        "-S",
        "--sds",
        dest="sds",
        action="store",
        metavar="filename string",
        default="DEFAULT",
        type=str,
        help="This is the filename of the time point standard devations of the data series you wish to analyze.\
                   The data should be tab-spaced. The first row should contain a # sign followed by the time points with either CT or ZT preceding the time point (such as ZT0 or ZT4). Longer or shorter prefixes will not work. The following rows should contain the gene/series ID followed by the values for every time point. Where values are not available NA should be put in it's place.",
    )

    analysis.add_argument(
        "-N",
        "--ns",
        dest="ns",
        action="store",
        metavar="filename string",
        default="DEFAULT",
        type=str,
        help="This is the filename of the time point replicate numbers of the data series you wish to analyze.                         \
                   The data should be tab-spaced. The first row should contain a # sign followed by the time points with either CT or ZT preceding the time point (such as ZT0 or ZT4). Longer or shorter prefixes will not work. The following rows should contain the gene/series ID followed by the values for every time point. Where values are not available NA should be put in it's place.",
    )

    analysis.add_argument(
        "-x",
        "--prefix",
        dest="prefix",
        type=str,
        metavar="string",
        action="store",
        default="",
        help="string to be inserted in the output filename for this run",
    )

    analysis.add_argument(
        "-r",
        "--reps",
        dest="reps",
        type=int,
        metavar="int",
        action="store",
        default=2,
        help="# of reps of each time point to bootstrap (1 or 2, generally)",
    )

    analysis.add_argument(
        "-z",
        "--size",
        dest="size",
        type=int,
        metavar="N",
        action="store",
        default=50,
        help="Number of bootstrap resamples per gene.",
    )

    analysis.add_argument(
        "-j",
        "--workers",
        dest="workers",
        type=int,
        metavar="N",
        action="store",
        default=1,
        help="Parallel worker processes. 0 = all available CPUs.",
    )

    analysis.add_argument(
        "-w",
        "--waveform",
        dest="waveform",
        type=str,
        metavar="filename string",
        action="store",
        default="cosine",
        # choices=["waveform_cosine.txt","waveform_rampup.txt","waveform_rampdown.txt","waveform_step.txt","waveform_impulse.txt","waveform_trough.txt"],
        help="Should be a file with waveforms  you wish to search for listed in a single column separated by newlines.\
                          Options include cosine (dflt), trough",
    )

    analysis.add_argument(
        "--width",
        "-a",
        "--asymmetry",
        dest="width",
        type=str,
        metavar="filename string",
        action="store",
        default="widths_02-22.txt",
        # choices=["widths_02-22.txt","widths_04-20_by4.txt","widths_04-12-20.txt","widths_08-16.txt","width_12.txt"]
        help='Should be a file with asymmetries (widths) you wish to search for listed in a single column separated by newlines.\
                          Provided files include files like "widths_02-22.txt","widths_04-20_by4.txt","widths_04-12-20.txt","widths_08-16.txt","width_12.txt"\nasymmetries=widths',
    )
    analysis.add_argument(
        "-s",
        "-ph",
        "--phase",
        dest="phase",
        metavar="filename string",
        type=str,
        default="phases_00-22_by2.txt",
        help='Should be a file with phases you wish to search for listed in a single column separated by newlines.\
                          Example files include "phases_00-22_by2.txt" or "phases_00-22_by4.txt" or "phases_00-20_by4.txt"',
    )

    analysis.add_argument(
        "-p",
        "--period",
        dest="period",
        metavar="filename string",
        type=str,
        action="store",
        default="period_24.txt",
        help='Should be a file with phases you wish to search for listed in a single column separated by newlines.\
                          Provided file is "period_24.txt"',
    )

    analysis.add_argument(
        "--vash",
        dest="vash",
        action="store_true",
        default=False,
        help="Determine if you would like to use limma or Vash",
    )

    analysis.add_argument(
        "-U",
        "--noreps",
        "--unique",
        dest="noreps",
        action="store_true",
        default=False,
        help="Determine if your data has no replicates and therefore the standard deviation should be estimated from the arrhythmic time series",
    )

    analysis.add_argument(
        "-R",
        "--rnaseq",
        dest="rnaseq",
        action="store_true",
        default=False,
        help="Flag for data that is RNA-Seq and for which voom should be used.",
    )

    analysis.add_argument(
        "-L",
        "--limma",
        dest="limma",
        action="store_true",
        default=False,
        help="Flag for using the limma variance shrinkage methods",
    )

    analysis.add_argument(
        "-J",
        "--jtk",
        dest="jtk",
        metavar="filename string",
        type=str,
        action="store",
        help="The eJTK file to use if you don't have replicates in in your time series. The standard deviation between points will be estimated based on the arrhythmic time series.",
    )

    analysis.add_argument(
        "-B",
        "--basic",
        dest="basic",
        action="store_true",
        default=False,
        help="Flag for not using the limma or vash settings",
    )

    analysis.add_argument(
        "-W",
        "--write",
        dest="write",
        action="store_true",
        default=False,
        help="If you want pickle output from BooteJTK, use this flag.",
    )

    distribution = analysis.add_mutually_exclusive_group(required=False)
    distribution.add_argument(
        "-e",
        "--exact",
        dest="harding",
        action="store_true",
        default=False,
        help="use Harding's exact null distribution (dflt)",
    )
    distribution.add_argument(
        "-g",
        "--gaussian",
        "--normal",
        dest="normal",
        action="store_true",
        default=False,
        help="use normal approximation to null distribution",
    )

    return p


def cli():
    parser = __create_parser__()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli()
