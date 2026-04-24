# CIRC — Claude Code Guidelines

## Build & dependency management

This project uses **Poetry** for dependency management and packaging.

- Install dependencies: `poetry install`
- Run tests: `poetry run pytest`
- Run a specific test: `poetry run pytest tests/path/to/test.py -v`
- Run a script / CLI: `poetry run circ <subcommand>`
- Add a dependency: `poetry add <package>`
- Add a dev dependency: `poetry add --group dev <package>`

**Never use bare `python3 -m pytest` or `pip install`.** Always prefix with `poetry run`.

## Project layout

```
circ/                        Main package
  __init__.py
  cli.py                     Unified `circ` CLI entry point
  io.py                      Shared I/O utilities (read_expression, write_expression, sidecar_path)
  simulations.py             Shared simulation class (3-class: circadian/linear/constitutive)
  pirs/                      Prediction Interval Ranking Score
  bootjtk/                   Bootstrap empirical JTK circadian detection
  limbr/                     KNN imputation + SVA batch-effect removal
  expression_classification/ Unified PIRS + BooteJTK classifier
tests/                       pytest test suite (mirrors circ/ layout)
  test_io.py                 Tests for circ.io utilities
```

## Data format

Expression inputs and outputs support two formats, detected by file extension:
- **`.parquet`** — Apache Parquet (preferred for data at rest; faster, smaller, dtype-safe)
- **anything else** — tab-separated text (legacy/interop)

Column convention for expression data:
- First column: gene/peptide ID (header cell is `#` or `ID`)
- Remaining columns: ZT- or CT-prefixed sample names, e.g. `ZT02_1`, `ZT04_2`

All module constructors (`ranker`, `sva`, `imputable`, `Classifier`) accept either a file path
or a `pd.DataFrame` directly.  Use `circ.io.read_expression` / `write_expression` for
format-transparent I/O in custom scripts.

## Key conventions

- All modules follow a class-based API (e.g. `ranker`, `sva`, `Classifier`).
- Pipeline outputs land alongside input files unless routed to a temp dir.
- BooteJTK ref files live in `circ/bootjtk/ref_files/`.
