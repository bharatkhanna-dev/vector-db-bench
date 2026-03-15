from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
import random
import time
from typing import Iterable


@dataclass(frozen=True)
class VectorRecord:
    record_id: str
    vector: tuple[float, ...]
    cluster: int


@dataclass(frozen=True)
class BenchmarkResult:
    backend: str
    recall_at_k: float
    p50_ms: float
    p95_ms: float
    build_ms: float
    candidate_ratio: float


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


class ExactLinearBackend:
    name = "exact-linear"

    def build(self, records: Iterable[VectorRecord]) -> None:
        self.records = list(records)

    def search(self, query: tuple[float, ...], top_k: int) -> tuple[list[str], int]:
        ranked = sorted(
            self.records,
            key=lambda record: cosine_similarity(query, record.vector),
            reverse=True,
        )[:top_k]
        return [record.record_id for record in ranked], len(self.records)


class SignBucketANNBackend:
    name = "sign-bucket-ann"

    def __init__(self, signature_dims: int = 4) -> None:
        self.signature_dims = signature_dims
        self.records: list[VectorRecord] = []
        self.buckets: dict[tuple[int, ...], list[VectorRecord]] = {}

    def _signature(self, vector: tuple[float, ...]) -> tuple[int, ...]:
        return tuple(1 if value >= 0 else 0 for value in vector[: self.signature_dims])

    def build(self, records: Iterable[VectorRecord]) -> None:
        self.records = list(records)
        self.buckets = {}
        for record in self.records:
            self.buckets.setdefault(self._signature(record.vector), []).append(record)

    def search(self, query: tuple[float, ...], top_k: int) -> tuple[list[str], int]:
        candidates = list(self.buckets.get(self._signature(query), []))
        if len(candidates) < top_k:
            candidates = list(self.records)
        ranked = sorted(
            candidates,
            key=lambda record: cosine_similarity(query, record.vector),
            reverse=True,
        )[:top_k]
        return [record.record_id for record in ranked], len(candidates)


class ProjectionANNBackend:
    name = "projection-ann"

    def __init__(self, candidate_pool: int = 18) -> None:
        self.candidate_pool = candidate_pool
        self.records: list[VectorRecord] = []

    def build(self, records: Iterable[VectorRecord]) -> None:
        self.records = list(records)

    def search(self, query: tuple[float, ...], top_k: int) -> tuple[list[str], int]:
        candidates = sorted(
            self.records,
            key=lambda record: abs(record.vector[0] - query[0]) + abs(record.vector[1] - query[1]),
        )[: self.candidate_pool]
        ranked = sorted(
            candidates,
            key=lambda record: cosine_similarity(query, record.vector),
            reverse=True,
        )[:top_k]
        return [record.record_id for record in ranked], len(candidates)


def generate_dataset(
    seed: int = 7,
    clusters: int = 4,
    points_per_cluster: int = 24,
    dimensions: int = 8,
) -> list[VectorRecord]:
    generator = random.Random(seed)
    centers = [
        tuple(generator.uniform(-1.0, 1.0) for _ in range(dimensions))
        for _ in range(clusters)
    ]
    records: list[VectorRecord] = []
    for cluster_id, center in enumerate(centers):
        for point_idx in range(points_per_cluster):
            vector = tuple(center[dim] + generator.uniform(-0.08, 0.08) for dim in range(dimensions))
            records.append(
                VectorRecord(
                    record_id=f"c{cluster_id}-p{point_idx}",
                    vector=vector,
                    cluster=cluster_id,
                )
            )
    return records


def build_queries(records: list[VectorRecord], per_cluster: int = 3, seed: int = 11) -> list[tuple[float, ...]]:
    generator = random.Random(seed)
    grouped: dict[int, list[VectorRecord]] = {}
    for record in records:
        grouped.setdefault(record.cluster, []).append(record)

    queries: list[tuple[float, ...]] = []
    for cluster_records in grouped.values():
        for record in cluster_records[:per_cluster]:
            queries.append(tuple(value + generator.uniform(-0.02, 0.02) for value in record.vector))
    return queries


def recall_at_k(reference: list[str], predicted: list[str]) -> float:
    if not reference:
        return 0.0
    return len(set(reference).intersection(predicted)) / len(reference)


def benchmark_backend(
    backend,
    records: list[VectorRecord],
    queries: list[tuple[float, ...]],
    reference_backend: ExactLinearBackend,
    top_k: int = 5,
) -> BenchmarkResult:
    build_start = time.perf_counter()
    backend.build(records)
    build_ms = (time.perf_counter() - build_start) * 1000

    latencies_ms: list[float] = []
    recalls: list[float] = []
    candidate_ratios: list[float] = []

    for query in queries:
        reference_ids, _ = reference_backend.search(query, top_k=top_k)
        search_start = time.perf_counter()
        predicted_ids, candidate_count = backend.search(query, top_k=top_k)
        latencies_ms.append((time.perf_counter() - search_start) * 1000)
        recalls.append(recall_at_k(reference_ids, predicted_ids))
        candidate_ratios.append(candidate_count / len(records))

    return BenchmarkResult(
        backend=backend.name,
        recall_at_k=round(sum(recalls) / len(recalls), 3),
        p50_ms=round(percentile(latencies_ms, 0.50), 4),
        p95_ms=round(percentile(latencies_ms, 0.95), 4),
        build_ms=round(build_ms, 4),
        candidate_ratio=round(sum(candidate_ratios) / len(candidate_ratios), 3),
    )


def run_benchmarks(top_k: int = 5) -> list[BenchmarkResult]:
    records = generate_dataset()
    queries = build_queries(records)
    reference_backend = ExactLinearBackend()
    reference_backend.build(records)
    backends = [reference_backend, SignBucketANNBackend(), ProjectionANNBackend()]
    return [benchmark_backend(backend, records, queries, reference_backend, top_k=top_k) for backend in backends]


def format_results_table(results: list[BenchmarkResult]) -> str:
    lines = [
        "| backend | recall@k | p50 (ms) | p95 (ms) | build (ms) | candidate ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            f"| {result.backend} | {result.recall_at_k:.3f} | {result.p50_ms:.4f} | {result.p95_ms:.4f} | {result.build_ms:.4f} | {result.candidate_ratio:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    results = run_benchmarks(top_k=5)
    print("Vector search benchmark summary\n")
    print(format_results_table(results))
