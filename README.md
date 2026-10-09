# Rössler Dataset Generation and Classification

MATLAB generators and a Python/CUDA interface for six-class classification of the modified Rössler system:

```text
dx/dt = -y - z
dy/dt = x + a*y
dz/dt = b*x + z*(x - c)
```

## Files

| File | Purpose |
| --- | --- |
| `create_rossler_dataset_afixed.m` | Generates a 100×100 training parameter grid with fixed `a = 0.35`. |
| `create_rossler_afixed_test.m` | Generates 200 test trajectories with independent random ±0.05 offsets in `b` and `c`, excluding training-grid and duplicate combinations. |
| `classify_rossler.py` | Single-sample classification API and batch command-line interface. |
| `rossler_classifier.cu` | DCP-derived symbolic sequence, periodicity, K-value, and LZ76 calculations, compiled at runtime. |
| `parameters.mat` | Training parameters `[index, a, b, c]`. |
| `test_samples.mat` | The 200 selected source indices for test generation. |

## Generate data

Run from this directory in MATLAB:

```matlab
create_rossler_dataset_afixed();
create_rossler_afixed_test;
```

Both generators use `dt = 0.005` and a default duration of 200 seconds. Training and test initial states are `[0,0,0.1]` and `[0,0,1]`. Each sample stores raw states and its generating parameters. Invalid or timed-out integrations are skipped. MATLAB Parallel Computing Toolbox is required.

## Classify one sample

Install NumPy and SciPy, and provide an NVIDIA GPU with CUDA/NVRTC available. A CUDA-enabled PyTorch installation can also supply NVRTC; alternatively set `NVRTC_PATH` to its library file.

```python
from classify_rossler import classify_rossler

label = classify_rossler("sample.mat")  # Integer class 1-6
label = classify_rossler(states, parameters=[a, b, c])
details = classify_rossler("sample.mat", return_details=True)
```

MAT samples should contain `y_solution` (N×3) and `a,b,c`. Default `system` mode recomputes a long orbit from those parameters. `mode="trajectory"` analyzes only the supplied states and requires at least 2000 detected symbols; short trajectories are rejected in this mode.

```bash
python classify_rossler.py --data sample.mat
python classify_rossler.py --legacy-grid
```

`--legacy-grid` reproduces the original training-grid arithmetic and must not be used for shifted test parameters. Batch output includes labels, K values, and an optional reference comparison. All 10000 historical training labels were independently reproduced on the tested CUDA setup.

The A–F mapping is specific to this dataset. It uses five periodic K signatures (`0`, `-2/3`, `-2/7`, `-18/31`, `-8/15`); F covers other valid signatures. It is not an intrinsic six-class definition from the paper.

## Reference

Malykh et al., *Homoclinic chaos in the Rössler model*, Chaos 30, 113126 (2020). [Paper](https://doi.org/10.1063/5.0026188) · [DCP source](https://bitbucket.org/pusuluri_krishna/deterministicchaosprospector/).
