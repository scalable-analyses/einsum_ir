# Examples

Runnable, illustrative workloads for the etops package.

| Directory | What it covers |
|-----------|----------------|
| [spec/permutation](spec/permutation/) | Unary copy / transpose schedules grouped per einsum family; programmatic ``_cases.py`` regressions for non-textual variants |
| [spec/tensor_contraction](spec/tensor_contraction/) | Binary-contraction schedules grouped per einsum family, including the batched-GEMM family |
| [contraction/tccg](contraction/tccg/) | Pairwise TCCG benchmark corpus (24 default + 48 full); verify-always, opt-in performance measurement |

Every artifact name follows ``<einsum_family>/<descriptor>``.
For ``.teir`` files the family is the subdirectory name and the descriptor is the filename;
for ``_cases.py`` entries the dict key uses the same shape and the builder sets the IR ``@name`` to ``<family>_<descriptor>`` (slash collapsed to underscore so the IR identifier is a single token).

### spec/permutation artifacts

| Artifact | Conceptual role | Einsum | Dtype |
|----------|-----------------|--------|-------|
| ``abcd_dcba/scalar.teir`` | 4-D reverse, scalar Copy | ``abcd -> dcba`` | f32 |
| ``abcd_dcba/tiled.teir``  | 4-D reverse, tile-Copy primitive | ``abcd -> dcba`` | f32 |
| ``behi_ehib/scalar.teir`` | 4-D rotation (leading axis to trailing) | ``[b,e,h,i] -> [e,h,i,b]`` | f32 |
| ``acdb_abcd/scalar.teir`` | 4-D partial sort (sort axis names) | ``[a,c,d,b] -> [a,b,c,d]`` | f32 |
| ``a_a/scalar_f64``        | f64 vector copy regression | ``a -> a`` | f64 |
| ``abc_cba/scalar_f64``    | 3-D reverse, f64 | ``[a,b,c] -> [c,b,a]`` | f64 |
| ``rank9_arbitrary/scalar``| 9-D arbitrary permutation | ``[d0..d8] -> [d2,d1,d4,d0,d5,d7,d3,d8,d6]`` | f32 |
| ``rank9_same_inner/scalar`` | 9-D permutation preserving innermost axis | ``[d0..d8] -> [d2,d1,d4,d0,d5,d7,d3,d6,d8]`` | f32 |
| ``abc_bca/strided_input`` | 3-D rotation with non-contiguous input strides (TPP backend skip) | ``[a,b,c] -> [b,c,a]`` | f32 |

### spec/tensor_contraction artifacts

| Artifact | Kernel shape | Einsum |
|----------|--------------|--------|
| ``trus_pqtu_pqrs/scalar.teir``  | Scalar Contraction (no M/N/K in primitive) | ``trus,pqtu -> pqrs`` |
| ``trus_pqtu_pqrs/gemm.teir``    | GEMM (1 K axis in primitive) | ``trus,pqtu -> pqrs`` |
| ``trus_pqtu_pqrs/brgemm.teir``  | BRGEMM (2 K axes in primitive) | ``trus,pqtu -> pqrs`` |
| ``abcd_efab_efcd/brgemm.teir``  | BRGEMM, K axes contiguous in in0 | ``abcd,efab -> efcd`` |
| ``acbd_eafb_ecfd/brgemm.teir``  | BRGEMM, K axes interleaved with M in in0 | ``acbd,eafb -> ecfd`` |
| ``dba_dac_dbc/scalar.teir``     | Batched GEMM, scalar Contraction with ``first(a)`` guard on Zero | ``dba,dac -> dbc`` |
| ``dba_dac_dbc/scalar_reordered.teir`` | Batched GEMM, Zero hoisted out of the K loop | ``dba,dac -> dbc`` |
| ``yxgcaei_yxhfca_yhgfxei/scalar`` | All-roles-populated rank-7/6/7 regression (C: y, x ; M: g, e, i ; N: h, f ; K: c, a) | ``yxgcaei,yxhfca -> yhgfxei`` |

BRGEMM-shaped artifacts skip the BLAS backend (cblas has no batch-reduce API).

## Running spec examples

From Python (defaults to the TPP backend and each example's default artifact):

```python
from examples.spec.tensor_contraction import run
run()                                              # tpp, default artifact
run(backend="blas")                                # blas
run(name="dba_dac_dbc/scalar")                     # parse and run the batched-GEMM scalar schedule
run(name="trus_pqtu_pqrs/brgemm", show=True)       # print the parsed IR tree first
```

From the command line (``--name`` accepts any ``<einsum_family>/<schedule>`` path):

```sh
python -m examples.spec.tensor_contraction
python -m examples.spec.tensor_contraction --backend blas
python -m examples.spec.tensor_contraction --name dba_dac_dbc/scalar
python -m examples.spec.tensor_contraction --name trus_pqtu_pqrs/brgemm --show
python -m examples.spec.permutation --name abcd_dcba/tiled
```

Each ``run()`` parses (or builds) the chosen artifact under the example directory,
applies a **minimal pre-lowering pipeline** consisting of only ``EnsureKernelShape`` (which synth-fills any empty M/N/K role lists so libxsmm/cblas can dispatch the contraction shape),
compiles for the requested backend (TPP by default; BLAS via the ``backend`` argument or ``--backend`` flag),
executes, and asserts numerical agreement against an independent NumPy reference.

## Running the TCCG corpus

```sh
python -m examples.contraction.tccg                          # verify default on both backends
python -m examples.contraction.tccg --setting full           # verify full on both backends
python -m examples.contraction.tccg --backend tpp -v         # per-contraction line
python -m examples.contraction.tccg --perf --csv tccg.csv    # verify + measure + write CSV
```

See [contraction/tccg/README.md](contraction/tccg/README.md) for performance protocol, CSV schema, and wall-time notes.
