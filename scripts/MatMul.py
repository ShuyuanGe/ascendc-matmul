"""Reference (golden) implementation of the matmul operator.

The kernel computes C = A @ B with fp16 inputs and fp32 accumulation, then
stores the result as fp32.  Keep this file in sync with the dtype policy in
main.asc (kDtypeA/B/C) and with kernel.asc.
"""

import numpy as np


def impl(a, b):
    # numpy promotes fp16 inputs to fp32 here, so this is an fp32 accumulation
    # of exactly-representable fp16 products - the same arithmetic the Cube unit
    # performs.  Only the summation order may differ.
    return np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float32)
