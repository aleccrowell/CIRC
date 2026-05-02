import os
import pytest
from circ.simulations import simulate


@pytest.fixture(scope="session")
def simulated_expression_file(tmp_path_factory):
    """Write a small simulated expression TSV and return its path."""
    tmp = tmp_path_factory.mktemp("expr")
    out = str(tmp / "sim_expr.txt")
    # Small dataset: 30 genes, 6 timepoints, 2 reps — fast enough for CI
    sim = simulate(
        tpoints=6, nrows=30, nreps=2, tpoint_space=4, pcirc=0.4, plin=0.2, rseed=42
    )
    sim.write_output(out_name=out)
    return out
