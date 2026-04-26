# circ.bootjtk — Bootstrap empirical JTK rhythm detection

BooteJTK detects circadian rhythms in expression data using a bootstrapped
extension of the JTK_CYCLE algorithm, providing empirical p-values for each
gene's rhythmicity.

## CLI

### `circ rhythm` and `circ rhythm-calcp`

These subcommands pass all arguments through to BooteJTK's own parsers.
Run `circ rhythm --help` or `circ rhythm-calcp --help` for the full flag list.

Quick example:

```bash
# Single-step rhythm detection
circ rhythm -f normalized.txt -x OUTPUT_PREFIX -r 2 -z 10

# Full pipeline with empirical p-value calculation
circ rhythm-calcp -f normalized.txt -x OUTPUT_PREFIX -r 2 -z 10
```

Key BooteJTK flags:

| Flag | Description |
|---|---|
| `-f FILE` | Input file |
| `-x PREFIX` | Output file prefix |
| `-r INT` | Number of bootstrap resampling iterations |
| `-z INT` | Number of permutation iterations for empirical p-values |
| `-p PERIODS` | Comma-separated list of periods to test (default: 24) |
| `-a ASYMMETRIES` | Asymmetry values to test |
| `-s PHASES` | Phase offsets to test |
