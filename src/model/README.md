# GEMM Performance Model

This module predicts GEMM execution time in seconds for various CPU architectures.

## Supported Architectures

Currently, three architectures have specialized performance models:
- **AMD Zen5**
- **Apple M4**
- **ARM Cortex-A76**

For all other architectures, a simple heuristic is used as a fallback.

All predictions are for **FP32** values.

## Usage

```bash
./bench_model <m> <n> <k> <trans_a> <trans_b> <model>
```

Parameters:
- `m`, `n`, `k`: Matrix dimensions (C = A × B, where A is M×K, B is K×N, C is M×N)
- `trans_a`: Transpose A matrix (0 = no, 1 = yes)
- `trans_b`: Transpose B matrix (0 = no, 1 = yes)
- `model`: Architecture model (zen5, m4, a76, or generic)

Examples:
```bash
./bench_model 64 48 64 0 0 zen5
./bench_model 128 128 128 0 1 m4
```
