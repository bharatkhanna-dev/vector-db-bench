from __future__ import annotations

from vector_db_bench.benchmark import ExactLinearBackend, SignBucketANNBackend, benchmark_backend, build_queries, format_results_table, generate_dataset, percentile, run_benchmarks


def test_exact_backend_has_perfect_recall_against_reference() -> None:
    records = generate_dataset()
    queries = build_queries(records)
    reference = ExactLinearBackend()
    reference.build(records)
    result = benchmark_backend(reference, records, queries, reference, top_k=5)
    assert result.recall_at_k == 1.0
    assert result.candidate_ratio == 1.0


def test_percentile_is_monotonic() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0.5) <= percentile(values, 0.95)


def test_approximate_backend_reports_partial_candidate_ratio() -> None:
    records = generate_dataset()
    queries = build_queries(records)
    reference = ExactLinearBackend()
    reference.build(records)
    approx = SignBucketANNBackend(signature_dims=4)
    result = benchmark_backend(approx, records, queries, reference, top_k=5)
    assert 0.0 < result.candidate_ratio <= 1.0
    assert 0.0 <= result.recall_at_k <= 1.0


def test_results_table_contains_backend_names() -> None:
    table = format_results_table(run_benchmarks(top_k=5))
    assert "exact-linear" in table
    assert "projection-ann" in table
