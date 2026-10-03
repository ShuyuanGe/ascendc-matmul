"""Single source of truth for the test cases.

Both gen_data.py (which writes the inputs/golden) and verify_result.py (which
checks the kernel output) import this table, so adding or resizing a case is a
one-line edit here.  Use ./run.sh <case_id> to run a single case.
"""

# rtol/atol are the per-element tolerances; tol is the allowed fraction of
# mismatching elements (0.0 = every element must be within tolerance).
CASES = [
    {"m": 256, "k": 256, "n": 256, "seed": 11, "rtol": 1e-3, "atol": 1e-3, "tol": 0.0},
    {"m": 512, "k": 512, "n": 512, "seed": 12, "rtol": 1e-3, "atol": 1e-3, "tol": 0.0},
    {"m": 128, "k": 1024, "n": 768, "seed": 13, "rtol": 1e-3, "atol": 1e-3, "tol": 0.0},
]
