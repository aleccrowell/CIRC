"""Run all CIRC example scripts in order and report pass/fail."""
import subprocess
import sys
import time
from pathlib import Path

scripts = sorted(Path(__file__).parent.glob("0*.py"))

passed, failed = [], []
for script in scripts:
    print(f"\n{'=' * 60}")
    print(f"  {script.name}")
    print('=' * 60)
    t0 = time.time()
    result = subprocess.run([sys.executable, str(script)], check=False)
    elapsed = time.time() - t0
    if result.returncode == 0:
        passed.append(script.name)
        print(f"  OK ({elapsed:.0f}s)")
    else:
        failed.append(script.name)
        print(f"  FAILED (exit {result.returncode}, {elapsed:.0f}s)")

print(f"\n{'=' * 60}")
print(f"  {len(passed)}/{len(scripts)} examples passed")
if failed:
    print("  Failed:")
    for f in failed:
        print(f"    {f}")
print('=' * 60)
sys.exit(1 if failed else 0)
