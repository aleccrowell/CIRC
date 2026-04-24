import sys
import os

# Ensure the project root is on the path so `circ` is importable when running
# tests directly from this directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
