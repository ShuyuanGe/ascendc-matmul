"""Generate inputs and golden output for every case in cases.py.

Run from the build directory (./run.sh does this for you):
    python3 ../scripts/gen_data.py

Layout produced, per case i:
    input/case{i}/a.bin        fp16 [M, K]
    input/case{i}/b.bin        fp16 [K, N]
    input/case{i}/dims.txt     "M K N"
    output/golden_case{i}/golden_c.bin   fp32 [M, N]
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cases import CASES
from MatMul import impl


def main():
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    for cid, case in enumerate(CASES):
        m, k, n = case["m"], case["k"], case["n"]
        rng = np.random.default_rng(case["seed"])

        a = rng.uniform(-1.0, 1.0, size=(m, k)).astype(np.float16)
        b = rng.uniform(-1.0, 1.0, size=(k, n)).astype(np.float16)
        golden = impl(a, b)

        case_in = os.path.join("input", f"case{cid}")
        case_out = os.path.join("output", f"golden_case{cid}")
        os.makedirs(case_in, exist_ok=True)
        os.makedirs(case_out, exist_ok=True)

        a.tofile(os.path.join(case_in, "a.bin"))
        b.tofile(os.path.join(case_in, "b.bin"))
        with open(os.path.join(case_in, "dims.txt"), "w") as f:
            f.write(f"{m} {k} {n}\n")
        golden.tofile(os.path.join(case_out, "golden_c.bin"))

        print(f"case{cid}: A[{m},{k}] x B[{k},{n}] -> C[{m},{n}]")

    print(f"generated {len(CASES)} case(s)")


if __name__ == "__main__":
    main()
