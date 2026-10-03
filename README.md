# matmul

Matrix multiplication `C = A @ B` on the Ascend NPU via the Ascend C
high-level Matmul API. The build system, host runner, tiling plumbing, test-data
generation and verification are all wired up — **`kernel.asc` is the file you
edit.**

## Operator spec

* `A` — shape `[M, K]`, fp16, row-major (ND)
* `B` — shape `[K, N]`, fp16, row-major (ND)
* `C` — shape `[M, N]`, fp32, `C = A @ B` with fp32 accumulation

`M`, `K`, `N` are resolved at runtime from `TensorGroupInfo`, so one binary
covers every case in `scripts/cases.py` without recompiling.

## Layout

```
matmul/
├── CMakeLists.txt        # ASC project, host C++ standard = C++14
├── main.asc              # host wrapper: ACL init, load inputs, write output
├── kernel.asc            # >>> the kernel you write (working baseline included)
├── data_utils.h          # file read/write helpers
├── run.sh                # build + gen data + run + verify
└── scripts/
    ├── cases.py          # test-case table (M, K, N, seed, tolerances)
    ├── MatMul.py         # numpy golden reference
    ├── gen_data.py       # writes inputs and golden output
    └── verify_result.py  # compares kernel output vs golden
```

## Build & run

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
./run.sh          # all cases
./run.sh 1        # a single case
```

`run.sh` configures and builds the project, generates data into `build/input`
and `build/output`, runs the kernel on the NPU, and verifies each case.

## Writing the kernel

`kernel.asc` is `#include`d by `main.asc`, so host and device code share one
translation unit. Do not add `main()`, include guards or `#pragma once`, and keep
`run_kernel` non-static — it is the fixed host entry point.

The contract with the harness:

* `run_kernel(a, info_a, b, info_b, c, info_c, availableCoreNum, stream)` is the
  only entry the host calls. Tensors are GM addresses; shapes and dtypes come
  from the `TensorGroupInfo` arguments.
* Launch device work with the `<<<blocks, lcn, stream>>>` call operator.
* Free every scratch buffer (`aclrtMalloc`) you allocate. The baseline frees its
  tiling and workspace buffers after draining the stream.

The shipped baseline builds a `TCubeTiling` on the host, copies it to GM, and
drives an `AscendC::Matmul` object per AI Core. Sensible first tuning steps:

1. Adjust the tiling strategy in `GenerateTiling()` (core split, base M/N/K, L1
   buffering, `SetFixSplit`, double buffering).
2. Set `kIsTransA` / `kIsTransB` (plus the matching tiling `SetXType` flags) to
   consume transposed inputs.
3. Swap the high-level object for a hand-written `Mmad`/`Fixpipe` pipeline if you
   need full control of L1/L0 traffic.

Anything that changes the precision policy (for example an fp16 output) must be
mirrored in `main.asc` (`kDtypeA/B/C`) and `scripts/MatMul.py`.

## Test cases

`scripts/cases.py` is the single source of truth. Add a case there and both
generation and verification pick it up automatically:

```python
{"m": 1024, "k": 1024, "n": 1024, "seed": 21, "rtol": 1e-3, "atol": 1e-3, "tol": 0.0},
```

## Notes

The target SoC defaults to `dav-2201` (Ascend 910B). Override it with
`-DCMAKE_ASC_ARCHITECTURES=<arch>` on the `cmake` line, or build an
unoptimised kernel with `-DENABLE_ASC_DEBUG=ON`.
