# circ.rhythmicity — Circadian rhythmicity detection and classification

`circ.rhythmicity` provides two complementary algorithms for detecting and
characterising circadian rhythms in expression data:

- **BooteJTK** — non-parametric, bootstrap-based rhythm detection using
  Kendall's τ against reference waveforms; robust to noise and missing data
- **ECHO** — parametric amplitude-aware fitting; classifies oscillations as
  *damped*, *harmonic*, or *forced* based on how amplitude changes over time

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
| `-r INT` | Replicates per timepoint |
| `-z INT` | Bootstrap iterations per gene |
| `-j INT` | Parallel worker processes (0 = all CPUs) |
| `-p FILE` | Periods to test (default: 24 h) |
| `-a FILE` | Asymmetry (waveform width) values |
| `-s FILE` | Phase offsets |

## Python API

### BooteJTK — `circ.rhythmicity.pipeline`

Used internally by `Classifier.run_bootjtk()`.  Can also be called directly:

```python
import types
from circ.rhythmicity.pipeline import main as pipeline_main

args = types.SimpleNamespace(
    filename="normalized.txt",
    output="DEFAULT",
    prefix="run",
    reps=2,
    size=50,
    workers=1,
    basic=True,
    ...
)
pipeline_main(args)
```

### ECHO — `circ.rhythmicity.echo_fit.EchoFitter`

Fits the ECHO amplitude-aware oscillator model per gene:

    x(t̂) = A · exp(−γt̂²) · cos(ωt̂ + φ) + y

where t̂ is time normalised to [0, 1].  The amplitude change coefficient γ
classifies each oscillation:

| Class | Condition | Interpretation |
|---|---|---|
| `damped` | γ > 0.03 | Amplitude decreases over the data window |
| `harmonic` | \|γ\| ≤ 0.03 | Approximately constant amplitude |
| `forced` | γ < −0.03 | Amplitude increases over the data window |

```python
from circ.rhythmicity.echo_fit import EchoFitter

# Accepts a file path (TSV or Parquet) or a DataFrame
fitter = EchoFitter("normalized.parquet", reps=2)
result = fitter.fit(workers=4)
```

`result` is a DataFrame indexed by gene ID with columns:

| Column | Description |
|---|---|
| `echo_A` | Fitted amplitude |
| `echo_gamma` | Amplitude change coefficient (normalised time) |
| `echo_period` | Fitted period (hours) |
| `echo_phase` | Fitted phase (hours) |
| `echo_baseline` | Fitted y-intercept |
| `echo_tau` | Kendall's τ goodness-of-fit |
| `echo_p` | p-value for τ |
| `echo_p_bh` | BH-corrected p-value |
| `echo_amplitude_class` | `'damped'`, `'harmonic'`, or `'forced'` |
| `echo_converged` | Whether the optimisation converged |

The most convenient way to run ECHO alongside BooteJTK is through
`Classifier.run_echo()` — see
[`circ/expression_classification/README.md`](../expression_classification/README.md).
