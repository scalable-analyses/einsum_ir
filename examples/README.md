# Examples

Runnable, illustrative workloads for the etops package.

| Directory | What it covers |
|-----------|----------------|
| [spec/permutation](spec/permutation/) | Scalar and tile-copy schedules for ``abcd -> dcba`` |
| [spec/batched_gemm](spec/batched_gemm/) | The ``dba,dac -> dbc`` batched matrix product |
| [spec/tensor_contraction](spec/tensor_contraction/) | The ``trus,pqtu -> pqrs`` contraction in three schedule shapes |

## Running

From Python (defaults to the TPP backend and each example's default schedule):

```python
from examples.spec.batched_gemm import run
run()                                # tpp, default schedule
run(backend="blas")                  # blas
run(schedule="scalar")               # parse and run scalar.teir
run(show=True)                       # print the parsed IR tree first
```

From the command line:

```sh
python -m examples.spec.batched_gemm
python -m examples.spec.batched_gemm --backend blas
python -m examples.spec.batched_gemm --schedule scalar
python -m examples.spec.batched_gemm --show
```

Each ``run()`` parses the chosen ``.teir`` file under the example
directory, applies a **minimal pre-lowering pipeline** consisting of
only ``EnsureKernelShape`` (which synth-fills any empty M/N/K role
lists so libxsmm/cblas can dispatch the contraction shape), compiles
for the requested backend (TPP by default; BLAS via the ``backend``
argument or ``--backend`` flag), executes, and asserts numerical
agreement against ``np.einsum`` / ``np.transpose``.