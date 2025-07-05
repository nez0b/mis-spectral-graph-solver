"""
Pytest configuration and fixtures specific to Dirac-3 quantum oracle tests.
"""

import pytest
import networkx as nx
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from motzkinstraus.oracles.dirac import DiracOracle
    DIRAC_AVAILABLE = True
except ImportError:
    DIRAC_AVAILABLE = False

try:
    from motzkinstraus.oracles.dirac_hybrid import DiracNetworkXHybridOracle
    DIRAC_HYBRID_AVAILABLE = True
except ImportError:
    DIRAC_HYBRID_AVAILABLE = False

try:
    from motzkinstraus.oracles.dirac_pgd_hybrid import DiracPGDHybridOracle
    DIRAC_PGD_HYBRID_AVAILABLE = True
except ImportError:
    DIRAC_PGD_HYBRID_AVAILABLE = False


# Pytest markers for Dirac tests
def pytest_configure(config):
    """Configure pytest markers for Dirac tests."""
    config.addinivalue_line("markers", "dirac: marks tests as requiring Dirac oracle")
    config.addinivalue_line("markers", "dirac_slow: marks tests as slow Dirac tests")
    config.addinivalue_line("markers", "dirac_cloud: marks tests requiring cloud connectivity")


@pytest.fixture
def dirac_oracle_basic():
    """Basic Dirac oracle for testing."""
    if not DIRAC_AVAILABLE:
        pytest.skip("Dirac oracle not available")
    return DiracOracle(num_samples=20, relax_schedule=1, solution_precision=0.001)


@pytest.fixture
def dirac_oracle_configs():
    """Different Dirac oracle configurations for testing."""
    if not DIRAC_AVAILABLE:
        pytest.skip("Dirac oracle not available")
    
    return [
        {"name": "Fast", "num_samples": 10, "relax_schedule": 1, "solution_precision": 0.01},
        {"name": "Balanced", "num_samples": 30, "relax_schedule": 2, "solution_precision": 0.001},
        {"name": "Precise", "num_samples": 50, "relax_schedule": 3, "solution_precision": 0.0001},
    ]


@pytest.fixture
def dirac_hybrid_oracle():
    """Dirac/NetworkX hybrid oracle."""
    if not DIRAC_HYBRID_AVAILABLE:
        pytest.skip("Dirac hybrid oracle not available")
    return DiracNetworkXHybridOracle(threshold_nodes=25)


@pytest.fixture
def dirac_pgd_hybrid_oracle():
    """Dirac/PGD hybrid oracle."""
    if not DIRAC_PGD_HYBRID_AVAILABLE:
        pytest.skip("Dirac PGD hybrid oracle not available")
    return DiracPGDHybridOracle(nx_threshold=25, verbose=False)


@pytest.fixture
def dirac_test_graphs():
    """Small graphs suitable for Dirac testing (limited by cloud costs)."""
    graphs = []
    
    # Very small graphs for Dirac testing
    G1 = nx.complete_graph(3)
    graphs.append(("Triangle K3", G1, 3))
    
    G2 = nx.cycle_graph(4)
    graphs.append(("4-cycle", G2, 2))
    
    G3 = nx.path_graph(5)
    graphs.append(("5-path", G3, 3))
    
    G4 = nx.complete_graph(4)
    graphs.append(("Complete K4", G4, 4))
    
    # Slightly larger graph for threshold testing
    G5 = nx.cycle_graph(8)
    graphs.append(("8-cycle", G5, 4))
    
    return graphs


@pytest.fixture
def dirac_performance_config():
    """Configuration for Dirac performance testing."""
    return {
        "timeout_seconds": 120,  # 2 minutes timeout for cloud tests
        "max_samples": 100,      # Limit samples to control costs
        "quick_config": {
            "num_samples": 15,
            "relax_schedule": 1,
            "solution_precision": 0.01
        },
        "standard_config": {
            "num_samples": 30,
            "relax_schedule": 2,
            "solution_precision": 0.001
        }
    }


@pytest.fixture
def dirac_comparison_graphs():
    """Graphs for comparing Dirac against other oracles."""
    return [
        (nx.cycle_graph(6), "6-cycle", 3),
        (nx.complete_graph(3), "Triangle", 3),
        (nx.path_graph(6), "6-path", 3),
        (nx.wheel_graph(5), "5-wheel", 2),  # 6 nodes total
    ]


# Custom pytest collection modifier for Dirac tests
def pytest_collection_modifyitems(config, items):
    """Automatically mark Dirac tests."""
    dirac_marker = pytest.mark.dirac
    dirac_slow_marker = pytest.mark.dirac_slow
    
    for item in items:
        # Mark all tests in dirac directory
        if "dirac" in str(item.fspath):
            item.add_marker(dirac_marker)
            
            # Mark slow tests
            if "large_graph" in item.name or "stress" in item.name or "performance" in item.name:
                item.add_marker(dirac_slow_marker)


# Timeout configuration for Dirac tests
@pytest.fixture(autouse=True)
def dirac_test_timeout(request):
    """Automatically apply longer timeouts for Dirac tests."""
    if request.node.get_closest_marker("dirac"):
        # Dirac tests get longer timeout due to cloud processing
        if not hasattr(request.node, "timeout"):
            request.node.timeout = 180  # 3 minutes default timeout