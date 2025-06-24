"""
Benchmarking framework for comparing Motzkin-Straus solvers with other MIS algorithms.
"""

from .networkx_comparison import (
    NetworkXComparisonBenchmark,
    BenchmarkResult,
    run_algorithm_comparison
)
from .graph_generators import (
    generate_test_graphs,
    GraphType,
    ScalingConfig
)
from .analysis import (
    analyze_benchmark_results,
    compute_approximation_ratios,
    statistical_summary
)

__all__ = [
    "NetworkXComparisonBenchmark",
    "BenchmarkResult", 
    "run_algorithm_comparison",
    "generate_test_graphs",
    "GraphType",
    "ScalingConfig",
    "analyze_benchmark_results",
    "compute_approximation_ratios",
    "statistical_summary"
]