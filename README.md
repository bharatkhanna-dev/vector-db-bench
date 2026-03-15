# Vector DB Benchmark Example

Public companion example for the project page: https://bharatkhanna.dev/projects/vector-db-bench/

Target standalone repository: https://github.com/bharatkhanna-dev/vector-db-bench

This example provides a deterministic local harness for comparing exact and approximate vector search strategies before you integrate real vendor SDKs.

## What is included

- `src/vector_db_bench/benchmark.py` — clustered dataset generation, backends, metrics, and reporting
- `tests/` — validation for recall, percentile calculations, and benchmark summaries
- `pyproject.toml` — standalone package metadata
- `LICENSE` — MIT license for the future dedicated repository

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m vector_db_bench
python -m pytest
```

## Why this example is useful

It makes benchmark methodology explicit:

- generate repeatable synthetic workloads,
- compare exact and approximate search behavior,
- inspect recall and latency together,
- and keep the harness easy to extend with real backend adapters later.

## Standalone repo readiness

This folder now has the basic structure required for promotion into its own repository:

- `src/` package layout,
- `pyproject.toml`,
- `.gitignore`,
- `LICENSE`,
- and isolated test configuration.
