"""Unified command-line interface for CIRC."""
import argparse
import sys


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------

def _run_impute(args):
    from circ.limbr.imputation import imputable
    obj = imputable(args.filename, missingness=args.missingness, neighbors=args.neighbors)
    obj.impute_data(args.output)


def _run_normalize(args):
    from circ.limbr.batch_fx import sva
    obj = sva(
        args.filename,
        design=args.design,
        data_type=args.data_type,
        blocks=args.blocks,
        pool=args.pool,
    )
    obj.preprocess_default()
    obj.perm_test(nperm=args.nperm, npr=args.nproc)
    obj.output_default(args.output)


def _run_rank(args):
    from circ.pirs.rank import ranker
    obj = ranker(args.filename, anova=not args.no_anova)
    sorted_data = obj.pirs_sort(outname=args.output)
    print(f"Ranked {len(sorted_data)} expression profiles → {args.output}")


# ---------------------------------------------------------------------------
# Parser builders
# ---------------------------------------------------------------------------

def _add_impute_parser(subparsers):
    p = subparsers.add_parser(
        "impute",
        help="Impute missing data with KNN (LIMBR)",
        description="Deduplicates peptides, drops high-missingness rows, and imputes remaining "
                    "missing values using K-nearest neighbours.",
    )
    p.add_argument("-f", "--filename", required=True, metavar="FILE",
                   help="Tab-separated input file (Peptide/Protein index + ZT columns)")
    p.add_argument("-m", "--missingness", type=float, default=0.3, metavar="FLOAT",
                   help="Maximum allowable missingness fraction per row (default: 0.3)")
    p.add_argument("-n", "--neighbors", type=int, default=10, metavar="N",
                   help="Number of nearest neighbours for KNN imputation (default: 10)")
    p.add_argument("-o", "--output", required=True, metavar="FILE",
                   help="Output file path")
    return p


def _add_normalize_parser(subparsers):
    p = subparsers.add_parser(
        "normalize",
        help="Normalize and remove batch effects (LIMBR SVA)",
        description="Applies pool normalization, quantile normalization, and SVA-based batch "
                    "effect identification and removal.",
    )
    p.add_argument("-f", "--filename", required=True, metavar="FILE",
                   help="Tab-separated input file (imputed proteomics or RNAseq)")
    p.add_argument("-d", "--design", required=True, choices=["c", "t", "b"],
                   help="Experimental design: c=circadian, t=timecourse, b=block")
    p.add_argument("-t", "--data-type", required=True, choices=["p", "r"], dest="data_type",
                   help="Data type: p=proteomics (Peptide/Protein index), r=RNAseq (# index)")
    p.add_argument("-b", "--blocks", default=None, metavar="FILE",
                   help="Parquet file with 'block' column (required when design=b)")
    p.add_argument("-p", "--pool", default=None, metavar="FILE",
                   help="Parquet file with 'pool_number' column (proteomics pool controls)")
    p.add_argument("--nperm", type=int, default=200, metavar="N",
                   help="Permutation iterations for batch effect significance (default: 200)")
    p.add_argument("--nproc", type=int, default=1, metavar="N",
                   help="Parallel worker processes for permutation testing (default: 1)")
    p.add_argument("-o", "--output", required=True, metavar="FILE",
                   help="Output file path")
    return p


def _add_rank_parser(subparsers):
    p = subparsers.add_parser(
        "rank",
        help="Rank expression profiles by constitutiveness (PIRS)",
        description="Sorts expression profiles from most to least constitutive using "
                    "prediction interval ranking scores.",
    )
    p.add_argument("-f", "--filename", required=True, metavar="FILE",
                   help="Tab-separated input file (# index + ZT columns)")
    p.add_argument("--no-anova", action="store_true",
                   help="Skip ANOVA pre-filtering of differentially expressed profiles")
    p.add_argument("-o", "--output", required=True, metavar="FILE",
                   help="Output scores file path")
    return p


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="circ",
        description="Circadian Integrated Research Core — unified analysis toolkit",
        epilog=(
            "rhythm and rhythm-calcp subcommands accept all standard BooteJTK flags.\n"
            "Run 'circ rhythm --help' or 'circ rhythm-calcp --help' for details."
        ),
    )
    parser.add_argument("--version", "-V", action="version",
                        version="%(prog)s " + _get_version())

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    _add_impute_parser(subparsers)
    _add_normalize_parser(subparsers)
    _add_rank_parser(subparsers)

    # rhythm and rhythm-calcp: capture remaining args and delegate to BooteJTK parsers
    rhythm_p = subparsers.add_parser(
        "rhythm",
        help="Bootstrap JTK circadian rhythm detection (BooteJTK)",
        add_help=False,
    )
    rhythm_p.add_argument("bootejtk_args", nargs=argparse.REMAINDER)

    rhythm_calcp_p = subparsers.add_parser(
        "rhythm-calcp",
        help="Bootstrap JTK + CalcP full circadian pipeline",
        add_help=False,
    )
    rhythm_calcp_p.add_argument("bootejtk_args", nargs=argparse.REMAINDER)

    if len(sys.argv) < 2:
        parser.print_help()
        return

    args = parser.parse_args()

    if args.command == "impute":
        _run_impute(args)
    elif args.command == "normalize":
        _run_normalize(args)
    elif args.command == "rank":
        _run_rank(args)
    elif args.command == "rhythm":
        sys.argv = [sys.argv[0]] + (args.bootejtk_args or [])
        from circ.bootjtk.BooteJTK import cli
        cli()
    elif args.command == "rhythm-calcp":
        sys.argv = [sys.argv[0]] + (args.bootejtk_args or [])
        from circ.bootjtk.pipeline import cli
        cli()
    else:
        parser.print_help()


def _get_version():
    from importlib.metadata import version, PackageNotFoundError
    try:
        return version("circ")
    except PackageNotFoundError:
        return "unknown"


if __name__ == "__main__":
    main()
