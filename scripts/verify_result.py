"""Compare kernel output against the golden reference for one case.

    python3 ../scripts/verify_result.py <case_id>

Reads output/c.bin (written by the kernel) and output/golden_c.bin (copied in
by run.sh from output/golden_case<case_id>/).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cases import CASES

try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = None


def verify_result(output_path, golden_path, dtype, rtol, atol, tol=0.0):
    output = np.fromfile(output_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)
    total_size = golden.size

    if output.size != golden.size:
        if output.size < golden.size:
            golden = golden[: output.size]
            print(f"WARNING: output {output.size} < golden {total_size} elements, truncated golden")
        else:
            print(f"FAILED: output has {output.size} elements, golden has {total_size} - size mismatch")
            return False

    missing = total_size - output.size

    if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
        errors = int(np.sum(output != golden)) + missing
        error_rate = errors / total_size if total_size > 0 else 0.0
        ok = (missing == 0 and np.array_equal(output, golden)) or error_rate <= tol
        tag = "PASSED" if ok else "FAILED"
        print(f"{tag}: {os.path.basename(output_path)} vs {os.path.basename(golden_path)}")
        print(f"  mismatched: {errors}/{total_size} ({error_rate * 100:.2f}%), tol={tol}")
        return ok

    cmp_output = output.astype(np.float32) if dtype == bfloat16 else output
    cmp_golden = golden.astype(np.float32) if dtype == bfloat16 else golden
    isclose = np.isclose(cmp_output, cmp_golden, rtol=rtol, atol=atol, equal_nan=True)
    errors = int(np.sum(~isclose)) + missing
    error_rate = errors / total_size if total_size > 0 else 0.0
    diff = np.abs(cmp_output - cmp_golden)
    if error_rate <= tol:
        print(f"PASSED: {os.path.basename(output_path)} vs {os.path.basename(golden_path)}")
        if errors > 0:
            print(f"  mismatched: {errors}/{total_size} ({error_rate * 100:.2f}%), tol={tol}")
        print(f"  max abs diff: {np.max(diff) if diff.size > 0 else 0.0:.3e}")
        return True
    print(f"FAILED: {os.path.basename(output_path)} vs {os.path.basename(golden_path)}")
    print(f"  mismatched: {errors}/{total_size} ({error_rate * 100:.2f}%), tol={tol}")
    print(f"  max abs diff: {np.max(diff) if diff.size > 0 else 0.0:.3e}")
    return False


def main():
    if len(sys.argv) < 2:
        print(f"usage: {os.path.basename(__file__)} <case_id>")
        return 1
    try:
        case_id = int(sys.argv[1])
    except ValueError:
        print(f"invalid case_id: {sys.argv[1]}")
        return 1
    if case_id not in range(len(CASES)):
        print(f"unknown case_id {case_id}. Available: {sorted(range(len(CASES)))}")
        return 1

    case = CASES[case_id]
    dtype = np.float32  # keep in sync with kDtypeC in main.asc
    return 0 if verify_result(
        os.path.join("output", "c.bin"),
        os.path.join("output", "golden_c.bin"),
        dtype,
        case["rtol"],
        case["atol"],
        case["tol"],
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
