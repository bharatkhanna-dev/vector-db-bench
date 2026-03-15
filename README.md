# vector-db-bench

Local benchmark harness for comparing exact vs approximate vector search. Deterministic synthetic data, pluggable backends, and a recall/latency/coverage report.

## Overview

Before evaluating vendor vector databases, build a benchmark harness you trust. This project generates reproducible synthetic datasets, runs queries through different search backends, and reports recall@k, latency percentiles, build time, and candidate coverage in a single table.

Three backends included: exact linear scan (baseline), sign-bucket ANN, and projection ANN. Each demonstrates a different accuracy-speed tradeoff. Designed to be extended with real vendor clients behind the same interface.

## Quick start

```bash
git clone https://github.com/bharatkhanna-dev/vector-db-bench.git
cd vector-db-bench
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e .[dev]
python -m vector_db_bench   # prints the comparison table
python -m pytest -v         # runs the test suite
```

Requires Python 3.11+.

## Project structure

```
src/vector_db_bench/
    benchmark.py    # dataset generation, backends, metrics, reporting
    __init__.py
    __main__.py
tests/
    test_benchmark.py   # recall, percentile, candidate ratio, output format
pyproject.toml
```

## Backends

**exact-linear** -- brute force cosine similarity. Recall is always 1.0. The reference for all comparisons.

**sign-bucket-ann** -- LSH-style bucketing on the sign pattern of the first N dimensions. Searches one bucket, falls back to full scan if the bucket is too small.

**projection-ann** -- two-stage retrieval: filter by L1 distance on first 2 dimensions, then rerank top candidates by cosine similarity. `candidate_pool` controls the tradeoff.

## Adding a backend

Implement `build(records)` and `search(query, top_k) -> (ids, candidate_count)`, then add it to the list in `run_benchmarks`:

```python
class MyBackend:
    name = "my-backend"

    def build(self, records):
        # index the records
        ...

    def search(self, query, top_k):
        # return (list of record IDs, number of candidates scanned)
        ...
```

## Sample output

```
| backend          | recall@k | p50 (ms) | p95 (ms) | build (ms) | candidate ratio |
| ---              |     ---: |     ---: |     ---: |       ---: |            ---: |
| exact-linear     |    1.000 |   0.2134 |   0.3521 |     0.0412 |           1.000 |
| sign-bucket-ann  |    0.917 |   0.0891 |   0.1234 |     0.0523 |           0.312 |
| projection-ann   |    0.850 |   0.0654 |   0.0987 |     0.0198 |           0.188 |
```

*(Numbers are illustrative -- actual values depend on your machine.)*

## Tests

```bash
python -m pytest -v
```

Covers: exact backend perfect recall, percentile monotonicity, ANN candidate ratio bounds, and report format validation.

## Write-up

Methodology and design notes: https://bharatkhanna.dev/projects/vector-db-bench/

## License

MIT
