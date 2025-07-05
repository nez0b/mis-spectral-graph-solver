# Dirac-3 Specific Tests

This directory contains tests that specifically require the Dirac-3 quantum computing oracle API. These tests are separated from the main test suite because:

1. **Conditional Dependencies**: The Dirac-3 API may not always be available in all environments
2. **External Service**: Dirac-3 tests depend on external quantum computing cloud services
3. **Longer Runtime**: Quantum computing tests typically take longer to execute
4. **Network Requirements**: Tests may require internet connectivity to access Dirac cloud services

## Test Files

- `test_dirac_hybrid.py` - Tests for Dirac/NetworkX hybrid solver
- `test_dirac_pgd_hybrid.py` - Tests for Dirac/PGD hybrid solver with high precision
- `test_oracle_comparison.py` - Direct comparison between JAX PGD and Dirac oracles
- `test_oracle_counting.py` - Oracle call counting validation with Dirac
- `test_hybrid_large_graph.py` - Hybrid oracle behavior on large graphs

## Running Dirac Tests

To run only the Dirac-specific tests:

```bash
pytest tests/dirac/ -v
```

To run all tests except Dirac tests:

```bash
pytest tests/ --ignore=tests/dirac/
```

To run Dirac tests with longer timeout (recommended):

```bash
pytest tests/dirac/ -v --timeout=300
```

## Test Skipping

All tests in this directory include proper `@pytest.mark.skipif` decorators that will automatically skip tests when:
- Dirac oracle dependencies are not installed
- Dirac cloud services are not accessible
- Required authentication/configuration is missing

## Performance Notes

Dirac tests may take significantly longer than other tests due to:
- Cloud service queue times
- Quantum annealing computation time
- Network latency

Consider using `@pytest.mark.slow` marker when running comprehensive test suites.