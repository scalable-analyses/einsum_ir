# TCCG benchmark driver

Pairwise tensor-contraction benchmark from the Tensor Contraction Code Generator (TCCG) of Springer and Bientinesi (see [Reference](#reference) below).
The corpus ships in two settings:

- `default` — 24 contractions exercising 3- to 7-axis pairwise products.
- `full` — 48 contractions, a superset of `default` adding axis-permutation variants of the same shape families.

The corpus literal lives in `__init__.py::CORPUS`;
each entry pairs an einsum string with an axis-extent map.
Wall-clock time for one full `--perf` run on both backends is on the order of minutes;
the `default` setting alone is dominated by the same entry.

## Verification (always)

Every contraction is emitted via `etops.emit.einsum`, compiled for each selected backend, executed, and the output is compared element-wise against `np.einsum(expr, ..., optimize=False)` (`atol=1e-4`, `rtol=1e-4`).
The exit code is non-zero on the first verification failure;
the failing `(contraction, backend)` pair is logged.

```sh
python -m examples.contraction.tccg                              # default on both backends
python -m examples.contraction.tccg --setting full               # full on both backends
python -m examples.contraction.tccg --backend tpp -v             # per-contraction line
```

## Performance (opt-in)

`--perf` adds a wall-time measurement after each verification (3 warmup + 10 measured runs by default; `--warmup`, `--measure`, `--stat` tune the protocol).
GFLOPS follows the standard convention:

```
GFLOPS = 2 × prod(output extents) × prod(K-axis extents) / execute_seconds / 1e9
```

`--csv PATH` (requires `--perf`) writes one row per `(contraction, backend)`.
Header:

```
contraction,extents,backend,dtype,compile_seconds,execute_median_seconds,execute_min_seconds,execute_gflops_median,warmup_runs,measure_runs
```

```sh
python -m examples.contraction.tccg --perf --csv /tmp/tccg.csv
```

## Reference

Paul Springer and Paolo Bientinesi. 2018.
Design of a High-Performance GEMM-like Tensor-Tensor Multiplication.
*ACM Trans. Math. Softw.* 44, 3, Article 28 (2018), 29 pages.
[doi:10.1145/3157733](https://doi.org/10.1145/3157733)
