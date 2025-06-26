# Test Suite Documentation

This document describes the comprehensive test suite for the Motzkin-Straus Maximum Independent Set (MIS) solver project.

## 📁 Test Directory Structure

```
tests/
├── conftest.py                           # General fixtures and configuration
├── dirac/                               # Dirac-3 quantum oracle specific tests
│   ├── conftest.py                      # Dirac-specific fixtures and markers
│   ├── README.md                        # Documentation for Dirac test separation
│   ├── test_dirac_hybrid.py            # Dirac/NetworkX hybrid solver tests
│   ├── test_dirac_pgd_hybrid.py         # Dirac/PGD hybrid precision tests
│   ├── test_oracle_comparison.py        # JAX PGD vs Dirac oracle comparison
│   ├── test_oracle_counting.py          # Oracle call counting with Dirac
│   └── test_hybrid_large_graph.py       # Large graph hybrid behavior tests
├── unit/                                # Unit tests for core functionality
│   ├── algorithms/                      # Core algorithm implementations
│   │   ├── test_clique_implementation.py    # MIS/clique algorithms & validation
│   │   ├── test_frank_wolfe.py             # Frank-Wolfe optimization oracle
│   │   ├── test_networkx_exact_debug.py    # NetworkX exact algorithm debugging
│   │   └── test_scipy_milp_demo.py         # SciPy vs Gurobi MILP comparison
│   └── oracles/                         # Oracle implementations
│       └── test_extended_oracle_comparison.py  # Multi-oracle comparison tests
├── integration/                         # Integration tests
│   └── hybrid/                          # Hybrid solver integration
│       └── test_hybrid_integration.py   # Benchmark framework integration
└── performance/                         # Performance and benchmarking tests
    ├── jax/                            # JAX-specific performance tests
    │   ├── test_metal_gpu_benchmark.py     # Metal GPU vs CPU benchmarks
    │   └── test_vmap_implementation.py     # JAX vmap batching validation
    └── scaling/                        # Scaling behavior tests
        └── test_large_graph_quick.py      # Large graph performance tests
```

## 🎯 Test Categories

### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual components in isolation
- **Dependencies**: Core project dependencies only (NetworkX, NumPy, etc.)
- **Runtime**: Fast (< 30 seconds total)
- **Examples**:
  - Algorithm correctness validation
  - Oracle basic functionality
  - MILP solver comparisons
  - Verification function testing

### Integration Tests (`tests/integration/`)
- **Purpose**: Test component interactions and framework integration
- **Dependencies**: May require optional dependencies (Gurobi, JAX)
- **Runtime**: Medium (30 seconds - 2 minutes)
- **Examples**:
  - Hybrid solver framework integration
  - Benchmark system end-to-end testing
  - Cross-oracle validation

### Performance Tests (`tests/performance/`)
- **Purpose**: Validate performance characteristics and scaling behavior
- **Dependencies**: JAX, optional GPU support
- **Runtime**: Variable (marked with `@pytest.mark.slow` for long tests)
- **Examples**:
  - JAX Metal GPU benchmarking
  - vmap batching performance
  - Large graph scaling tests

### Dirac Tests (`tests/dirac/`)
- **Purpose**: Test Dirac-3 quantum computing oracle integration
- **Dependencies**: Dirac-3 API, cloud connectivity
- **Runtime**: Slow (cloud processing times)
- **Special Notes**: Separated due to external dependencies and costs

## 🚀 Running Tests

### Basic Usage

```bash
# Run all tests (excluding Dirac)
pytest tests/ --ignore=tests/dirac/

# Run specific test categories
pytest tests/unit/                    # Unit tests only
pytest tests/integration/             # Integration tests only
pytest tests/performance/             # Performance tests only

# Run Dirac tests (requires cloud access)
pytest tests/dirac/
```

### Advanced Usage

```bash
# Fast development testing (core functionality)
pytest tests/unit/ -v

# Full test suite with verbose output
pytest tests/ --ignore=tests/dirac/ -v

# Run specific test files
pytest tests/unit/algorithms/test_clique_implementation.py -v

# Run tests matching a pattern
pytest -k "oracle" tests/              # All oracle-related tests
pytest -k "gurobi" tests/              # All Gurobi-related tests

# Run with specific markers
pytest tests/ -m "not slow"            # Skip slow tests
pytest tests/performance/ -m "slow"    # Only slow performance tests

# Parallel execution (if pytest-xdist installed)
pytest tests/unit/ -n auto
```

### Test Markers

The test suite uses several pytest markers for organization:

- `@pytest.mark.slow` - Tests that take longer than 30 seconds
- `@pytest.mark.skipif` - Conditional test skipping based on dependencies
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.performance` - Performance benchmarks
- `@pytest.mark.dirac` - Dirac-3 specific tests (auto-applied)
- `@pytest.mark.dirac_slow` - Slow Dirac tests (auto-applied)

## 🔧 Dependencies and Requirements

### Core Dependencies (Required)
- `pytest` - Test framework
- `networkx` - Graph algorithms
- `numpy` - Numerical computing
- `matplotlib` - Plotting (for visualization tests)

### Optional Dependencies
- `gurobipy` - Gurobi optimization solver
- `scipy` - SciPy MILP solver
- `jax` - JAX optimization oracles
- `jax[metal]` - Metal GPU support (macOS)

### Dirac Dependencies (External Service)
- Dirac-3 API access and authentication
- Internet connectivity for cloud services
- Valid Dirac quantum computing account

## 📊 Test Coverage

The test suite provides comprehensive coverage across:

### Algorithms
- ✅ Brute force MIS algorithms
- ✅ Oracle-based MIS algorithms  
- ✅ Motzkin-Straus theorem implementation
- ✅ Verification functions (clique, independent set)

### Oracles
- ✅ JAX Projected Gradient Descent (PGD)
- ✅ JAX Mirror Descent
- ✅ JAX Frank-Wolfe optimization
- ✅ Gurobi quadratic programming
- ✅ Dirac-3 quantum annealing (separate)
- ✅ Hybrid oracle combinations

### Solvers
- ✅ Gurobi MILP solver
- ✅ SciPy MILP solver
- ✅ NetworkX exact algorithms
- ✅ Cross-solver validation

### Performance
- ✅ JAX Metal GPU benchmarking
- ✅ vmap batching optimization
- ✅ Scaling behavior validation
- ✅ Oracle call counting

## 🎮 Configuration

### pytest.ini / pyproject.toml
The project uses `pyproject.toml` for pytest configuration:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "performance: marks tests as performance tests",
    "dirac: marks tests as requiring Dirac oracle",
]
```

### Environment Variables
Some tests may use environment variables:

```bash
# Optional: Gurobi license configuration
export GUROBI_HOME="/path/to/gurobi"
export GRB_LICENSE_FILE="/path/to/gurobi.lic"

# Optional: JAX configuration
export JAX_PLATFORM_NAME="metal"  # For Metal GPU on macOS
export JAX_ENABLE_X64=true        # For 64-bit precision
```

## 🔍 Test Data and Fixtures

### Common Fixtures (conftest.py)
- `small_test_graphs` - Collection of small graphs with known properties
- `test_output_dir` - Directory for test output files (`figures/tests/`)
- `oracle_configs` - Standard oracle configurations
- `benchmark_config` - Benchmark framework settings

### Dirac-Specific Fixtures (dirac/conftest.py)
- `dirac_oracle_basic` - Basic Dirac oracle instance
- `dirac_oracle_configs` - Various Dirac configurations
- `dirac_test_graphs` - Small graphs suitable for cloud testing
- `dirac_performance_config` - Cloud testing configurations

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# If you see import errors, ensure the environment is activated
source .venv/bin/activate

# And the package is installed in development mode
pip install -e .
```

**2. Missing Dependencies**
```bash
# Install optional dependencies as needed
uv add gurobipy          # For Gurobi tests
uv add "jax[metal]"      # For JAX Metal GPU tests
uv add scipy             # For SciPy MILP tests
```

**3. Dirac Test Failures**
```bash
# Dirac tests may fail without proper API access
# Skip Dirac tests during development:
pytest tests/ --ignore=tests/dirac/
```

**4. Slow Test Performance**
```bash
# Skip slow tests for faster development cycles
pytest tests/ -m "not slow"

# Run tests in parallel (if pytest-xdist installed)
pytest tests/ -n auto
```

### Test Output
- **Logs**: Test output includes detailed solver information
- **Plots**: Visualization tests save plots to `figures/tests/`
- **Performance**: Timing information included in test output

## 📈 Continuous Integration

For CI/CD pipelines, recommended test commands:

```bash
# Fast CI tests (core functionality)
pytest tests/unit/ tests/integration/ -v --tb=short

# Full CI tests (excluding external services)
pytest tests/ --ignore=tests/dirac/ -v --tb=short

# Performance CI tests (with timeout)
pytest tests/performance/ -v --tb=short --timeout=300

# Nightly tests (including Dirac if configured)
pytest tests/ -v --tb=short --timeout=600
```

## 📝 Contributing Tests

When adding new tests:

1. **Choose the right category**: Unit, integration, performance, or Dirac
2. **Use appropriate fixtures**: Leverage existing fixtures from conftest.py
3. **Add proper markers**: Use `@pytest.mark.slow` for long tests
4. **Handle dependencies**: Use `@pytest.mark.skipif` for optional dependencies
5. **Update documentation**: Add new test files to this documentation

### Test Naming Conventions
- Test files: `test_*.py`
- Test classes: `Test*` (PascalCase)
- Test functions: `test_*` (snake_case)
- Fixtures: descriptive names in snake_case

### Example Test Structure
```python
import pytest
import networkx as nx

class TestNewFeature:
    """Test new feature implementation."""
    
    @pytest.fixture
    def test_data(self):
        """Fixture providing test data."""
        return nx.cycle_graph(5)
    
    def test_basic_functionality(self, test_data):
        """Test basic functionality."""
        # Test implementation
        assert True
    
    @pytest.mark.slow
    def test_performance(self, test_data):
        """Test performance characteristics."""
        # Performance test implementation
        assert True
```

## 📚 Additional Resources

- **Project Documentation**: See `doc/` directory for algorithm details
- **API Documentation**: Docstrings in source code
- **Examples**: See `examples/` directory for usage examples
- **Benchmark Reports**: Generated in `figures/` directory

For questions or issues with the test suite, please check the project's issue tracker or documentation.