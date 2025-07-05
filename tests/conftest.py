"""
Pytest configuration and common fixtures for the test suite.
"""

import pytest
import networkx as nx
import numpy as np
from pathlib import Path
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def test_output_dir():
    """Fixture providing the test output directory for plots."""
    output_dir = Path("figures/tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def small_test_graphs():
    """Fixture providing a collection of small test graphs for validation."""
    graphs = []
    
    # Triangle (K3) - clique size 3, MIS size 1
    G1 = nx.complete_graph(3)
    graphs.append(("Triangle K3", G1, {"clique_size": 3, "mis_size": 1}))
    
    # Square (4-cycle) - clique size 2, MIS size 2
    G2 = nx.cycle_graph(4)
    graphs.append(("4-cycle", G2, {"clique_size": 2, "mis_size": 2}))
    
    # Path of 4 nodes - clique size 2, MIS size 2
    G3 = nx.path_graph(4)
    graphs.append(("4-path", G3, {"clique_size": 2, "mis_size": 2}))
    
    # Complete graph K4 - clique size 4, MIS size 1
    G4 = nx.complete_graph(4)
    graphs.append(("Complete K4", G4, {"clique_size": 4, "mis_size": 1}))
    
    # Star graph (5 nodes) - clique size 2, MIS size 4
    G5 = nx.star_graph(4)  # 5 nodes total
    graphs.append(("Star 5 nodes", G5, {"clique_size": 2, "mis_size": 4}))
    
    return graphs


@pytest.fixture
def medium_test_graphs():
    """Fixture providing medium-sized test graphs."""
    graphs = []
    
    # Petersen graph
    G1 = nx.petersen_graph()
    graphs.append(("Petersen", G1))
    
    # Wheel graph
    G2 = nx.wheel_graph(8)
    graphs.append(("Wheel 8", G2))
    
    # Grid graph
    G3 = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3))
    graphs.append(("Grid 3x3", G3))
    
    # Random graph
    G4 = nx.erdos_renyi_graph(10, 0.3, seed=42)
    graphs.append(("Random G(10,0.3)", G4))
    
    return graphs


@pytest.fixture
def large_test_graphs():
    """Fixture providing larger test graphs for performance testing."""
    graphs = []
    
    # Barabási-Albert graph
    G1 = nx.barabasi_albert_graph(20, 3, seed=42)
    graphs.append(("BA(20,3)", G1))
    
    # Dense random graph
    G2 = nx.erdos_renyi_graph(15, 0.5, seed=123)
    graphs.append(("Dense Random G(15,0.5)", G2))
    
    # Larger grid
    G3 = nx.convert_node_labels_to_integers(nx.grid_2d_graph(4, 5))
    graphs.append(("Grid 4x5", G3))
    
    return graphs


@pytest.fixture
def jax_oracle_config():
    """Fixture providing standard JAX oracle configuration for testing."""
    return {
        'learning_rate_pgd': 0.02,
        'learning_rate_md': 0.01,
        'max_iterations': 1000,
        'num_restarts': 3,
        'tolerance': 1e-6,
        'verbose': False
    }


@pytest.fixture
def dirac_oracle_config():
    """Fixture providing standard Dirac oracle configuration for testing."""
    return {
        'num_samples': 20,
        'relax_schedule': 2,
        'solution_precision': 0.001,
        'verbose': False
    }


@pytest.fixture
def benchmark_config():
    """Fixture providing standard benchmark configuration for testing."""
    return {
        'num_random_runs': 5,
        'fast_timeout': 10.0,
        'medium_timeout': 30.0,
        'slow_timeout': 60.0
    }


# Availability check fixtures
@pytest.fixture
def check_gurobi_available():
    """Fixture to check if Gurobi is available."""
    try:
        from motzkinstraus import GUROBI_AVAILABLE
        return GUROBI_AVAILABLE
    except ImportError:
        return False


@pytest.fixture
def check_dirac_available():
    """Fixture to check if Dirac is available."""
    try:
        from motzkinstraus.oracles.dirac import DiracOracle
        return True
    except ImportError:
        return False


@pytest.fixture
def check_scipy_milp_available():
    """Fixture to check if SciPy MILP is available."""
    try:
        from motzkinstraus import SCIPY_MILP_AVAILABLE
        return SCIPY_MILP_AVAILABLE
    except ImportError:
        return False


# Skip markers for conditional tests
pytest_marks = {
    'requires_gurobi': pytest.mark.skipif(
        not pytest.importorskip("motzkinstraus", reason="Cannot import motzkinstraus").GUROBI_AVAILABLE,
        reason="Gurobi not available"
    ),
    'requires_dirac': pytest.mark.skipif(
        True,  # This will be replaced with actual check
        reason="Dirac not available"
    ),
    'requires_metal_gpu': pytest.mark.skipif(
        os.environ.get('JAX_PLATFORM_NAME', '') != 'METAL',
        reason="Metal GPU not available"
    )
}


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "requires_gurobi: mark test as requiring Gurobi")
    config.addinivalue_line("markers", "requires_dirac: mark test as requiring Dirac")
    config.addinivalue_line("markers", "requires_metal_gpu: mark test as requiring Metal GPU")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "performance: mark test as performance benchmark")
    config.addinivalue_line("markers", "integration: mark test as integration test")