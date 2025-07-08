# Regularization Implementation for Motzkin-Straus Maximum Clique Computation

## Overview

This document describes the comprehensive regularization implementation added to the Motzkin-Straus codebase. The regularization framework transforms the standard Motzkin-Straus formulation from `x^T A x` to `x^T (A + cI) x`, where `c` is the regularization parameter and `I` is the identity matrix.

## Benefits of Regularization

- **Eliminates spurious solutions**: Removes non-clique local optima that can mislead optimization
- **Ensures one-to-one correspondence**: Creates clean mapping between optimal solutions and maximum cliques
- **Makes optimization landscape strictly concave**: Improves convergence properties
- **More robust convergence**: Provides more reliable results across different graph types

## Mathematical Foundation

### Standard Motzkin-Straus Formulation
```
max x^T A x  subject to  sum(x_i) = 1, x_i ≥ 0
```
where the optimal value relates to clique size ω as: `max(x^T A x) = 1 - 1/ω`

### Regularized Motzkin-Straus Formulation
```
max x^T (A + cI) x  subject to  sum(x_i) = 1, x_i ≥ 0
```
where the optimal value relates to clique size ω as: `max(x^T (A + cI) x) = 1 - (1-c)/ω`

### Important: Matrix vs Polynomial Representation
**Matrix form** (used by PGD oracles): `max(x^T (A + cI) x) = 1 - (1-c)/ω`

**Polynomial form** (used by Dirac submission): The polynomial representation uses coefficient 1.0 for each edge term [i,j], which gives exactly **half** the matrix form for the unregularized part:
- Matrix: `max(x^T A x) = (ω-1)/ω`  
- Polynomial: `max(Σ edges x_i x_j) = (ω-1)/(2ω)`
- Therefore: `max(polynomial) = (ω-1)/(2ω) + c/ω`
- Energy conversion: `ω = (1-2c)/(1 + 2×energy)` (for polynomial representation)

### Parameter Range
- **c ∈ [0, 1]**: Valid theoretical range
- **c = 0.0**: No regularization (equivalent to standard Motzkin-Straus)
- **c = 0.5**: Default recommended value
- **c = 1.0**: Maximum regularization

## Code Structure Changes

### New Oracle Components

#### 1. Regularization Base Classes (`src/motzkinstraus/oracles/regularized_base.py`)

**Core Classes Added:**
- `RegularizationFunction`: Abstract base for extensible regularization methods
- `IdentityRegularization`: Implements `A + cI` regularization
- `RegularizedOracle`: Base class for regularized oracle implementations

**Key Features:**
- Extensible design supporting future regularization methods beyond identity
- Correct omega conversion formulas for regularized objectives
- Special case handling for c=0 and c=1
- Comprehensive parameter validation

**Example Usage:**
```python
from motzkinstraus.oracles.regularized_base import IdentityRegularization

# Create regularization function
reg_func = IdentityRegularization(c=0.5)

# Apply to adjacency matrix
regularized_matrix = reg_func.apply(adjacency_matrix)
```

#### 2. Regularized JAX PGD Oracle (`src/motzkinstraus/oracles/jax_pgd_regularized.py`)

**New Oracle Implementation:**
- `RegularizedJAXPGDOracle`: Combines regularization with JAX Projected Gradient Descent
- Inherits from `RegularizedOracle` base class
- Supports all JAX PGD configuration options
- Includes comparison methods for analyzing regularization effects

**Factory Function:**
```python
from motzkinstraus.oracles.jax_pgd_regularized import create_regularized_jax_pgd_oracle

oracle = create_regularized_jax_pgd_oracle(
    c=0.5,                    # Regularization parameter
    learning_rate=0.02,       # PGD learning rate
    max_iterations=1000,      # Maximum iterations
    num_restarts=10,          # Number of random restarts
    verbose=False             # Suppress detailed output
)
```

### Updated Base Oracle (`src/motzkinstraus/oracles/base.py`)

**Backward-Compatible Enhancement:**
- Added optional `regularization_c` parameter to `get_omega()` method
- Maintains full backward compatibility with existing code
- Enables regularization in standard oracles when parameter is provided

### Solver Framework Integration (`src/motzkinstraus/solvers/omega.py`)

**New Solver Added:**
- `RegularizedJAXPGDOmegaSolver`: Integrates regularized oracle into solver framework
- Supports all standard solver interface methods
- Configurable regularization parameter via solver configuration

**Usage in Solver Framework:**
```python
OMEGA_SOLVERS = [
    "jax_pgd_regularized",    # NEW: Regularized JAX PGD Oracle
    "jax_pgd",                # Standard JAX PGD Oracle
    "networkx_exact",         # NetworkX exact solver
]
```

## New Scripts and Tools

### 1. Regularized Graph-to-Omega Script (`scripts/regularized_graph_to_omega.py`)

**Complete Workflow Implementation:**
- Supports both DIMACS and QPLIB input formats
- Applies regularization to polynomial terms before QCI submission
- Direct integration with Dirac-3 quantum annealer via QCI client
- Comprehensive error handling and validation

**Usage Examples:**
```bash
# Basic usage with default c=0.5
python scripts/regularized_graph_to_omega.py DIMACS/triangle.dimacs

# Custom regularization parameter
python scripts/regularized_graph_to_omega.py DIMACS/graph.dimacs --regularization-c 0.8

# High-sampling run with raw data saving
python scripts/regularized_graph_to_omega.py input.dimacs --regularization-c 0.3 --num-samples 500 --save-raw

# JSON output format
python scripts/regularized_graph_to_omega.py input.json --format json
```

**Command-Line Options:**
- `--regularization-c`: Regularization parameter [0, 1] (default: 0.5)
- `--num-samples`: Number of Dirac samples (default: 100)
- `--relax-schedule`: Dirac relaxation schedule 1-4 (default: 2)
- `--solution-precision`: Solution precision (optional)
- `--format`: Output format (table/json)
- `--save-raw`: Save raw Dirac response data
- `--show-theory`: Show theoretical omega lines in histograms
- `--job-name`: Custom job name for Dirac submission

### 2. PGD Testing Framework (`examples/test_regularized_pgd.py`)

**Comprehensive PGD Method Testing:**
- Focuses on gradient-based optimization methods
- Removes dependencies on Gurobi/SciPy MILP solvers
- Includes regularized vs. unregularized comparisons
- Supports multiple output formats (table/JSON/CSV)

**Solver Coverage:**
- JAX Projected Gradient Descent Oracle (standard)
- JAX Projected Gradient Descent Oracle (regularized) 
- JAX Mirror Descent Oracle
- NetworkX exact solver (for validation)

**Usage Examples:**
```bash
# Test all PGD methods
python examples/test_regularized_pgd.py DIMACS/triangle.dimacs

# Compare regularization effects
python examples/test_regularized_pgd.py DIMACS/graph.dimacs --compare-regularization

# Custom regularization parameter
python examples/test_regularized_pgd.py DIMACS/graph.dimacs --regularization-c 0.8

# JSON output for automated analysis
python examples/test_regularized_pgd.py DIMACS/graph.dimacs --format json --quiet
```

**Regularization Comparison Feature:**
- Tests multiple c values: [0.0, 0.3, 0.5, 1.0]
- Compares runtime and accuracy across regularization levels
- Validates c=0.0 equivalence to unregularized methods
- Provides statistical analysis of optimization consistency

## Test Suite Enhancements

### New Test Module (`tests/test_regularization.py`)

**Comprehensive Test Coverage:**

#### 1. Identity Regularization Tests (`TestIdentityRegularization`)
- Parameter validation for range [0, 1]
- Matrix application correctness
- Property preservation (symmetry, diagonal modification)
- Edge case handling (c=0, c=1)
- Invalid input rejection

#### 2. Parameter Validation Tests (`TestRegularizationValidation`)
- Range validation [0, 1]
- Type conversion and error handling
- Edge case acceptance (c=0.0, c=1.0)
- Invalid value rejection
- Convenience function testing

#### 3. Regularized Oracle Tests (`TestRegularizedJAXPGDOracle`)
- Oracle initialization and configuration
- Omega computation accuracy on known graphs
- Regularized vs. unregularized performance comparison
- Cherry graph spurious solution handling
- Optimization characteristics improvement verification

#### 4. Integration Tests (`TestRegularizationIntegration`)
- Consistency between regularization approaches
- Parameter sensitivity analysis across c values
- Empty and trivial graph handling
- Cross-validation with exact solvers

**Test Graphs Used:**
- Triangle graph (K₃): Maximum clique size 3
- Complete graphs (K₄): Maximum clique size 4
- Path graphs: Maximum clique size 2
- Cycle graphs: Maximum clique size 2
- Cherry graph: Known spurious solution case
- Disconnected graphs: Multiple components

**Validation Benchmarks:**
- Performance comparison across c values [0.0, 0.3, 0.5, 1.0]
- Energy variance reduction analysis
- Solution vector validation (simplex constraints)
- Runtime performance measurement

## Usage Guidelines

### Basic Regularized Computation

```python
# Import regularized oracle
from motzkinstraus.oracles.jax_pgd_regularized import create_regularized_jax_pgd_oracle
import networkx as nx

# Create graph
graph = nx.complete_graph(4)

# Create regularized oracle
oracle = create_regularized_jax_pgd_oracle(c=0.5)

# Compute maximum clique size
omega = oracle.get_omega(graph)
print(f"Maximum clique size: {omega}")
```

### Comparing Regularization Effects

```python
# Create oracles with different regularization
oracle_unreg = ProjectedGradientDescentOracle()
oracle_reg = create_regularized_jax_pgd_oracle(c=0.5)

# Compare on same graph
omega_unreg = oracle_unreg.get_omega(graph)
omega_reg = oracle_reg.get_omega(graph)

print(f"Unregularized: {omega_unreg}, Regularized: {omega_reg}")
```

### Solver Framework Integration

```python
from motzkinstraus.solvers.omega import create_omega_solvers

# Configure solvers including regularized
solvers = create_omega_solvers(
    ["jax_pgd", "jax_pgd_regularized", "networkx_exact"],
    {
        'jax_config': {
            'regularization_c': 0.5,
            'learning_rate_pgd': 0.02,
            'max_iterations': 1000,
            'num_restarts': 10
        },
        'networkx_config': {'max_nodes': 20}
    }
)

# Run all solvers
for solver in solvers:
    omega, runtime, success, error = solver.compute_omega(graph)
    print(f"{solver.name}: ω={omega}, time={runtime:.3f}s")
```

## Performance Characteristics

### Regularization Benefits Observed

1. **Improved Convergence Consistency**: Regularized methods show lower variance in optimization results
2. **Faster Convergence**: Often converges faster than unregularized versions
3. **Spurious Solution Elimination**: More reliable results on graphs with known problematic cases
4. **Parameter Robustness**: Results stable across reasonable c values (0.3-0.8)

### Recommended Parameter Values

- **c = 0.0**: Use when you need exact equivalence to standard Motzkin-Straus
- **c = 0.5**: Default recommended value for general use
- **c = 0.8**: Higher regularization for difficult graphs with many spurious solutions
- **c = 1.0**: Maximum regularization (may over-constrain some problems)

## Future Extensions

The regularization framework is designed for extensibility:

1. **Additional Regularization Functions**: Beyond identity matrix (A + cI)
2. **Adaptive Regularization**: Dynamic c parameter adjustment during optimization
3. **Problem-Specific Regularization**: Custom regularization for specific graph classes
4. **Multi-Parameter Regularization**: More complex regularization matrices

## Validation and Testing

### Equivalence Verification
- ✅ c=0 produces identical results to unregularized version
- ✅ All c ∈ [0,1] values produce valid omega results
- ✅ Edge cases (c=0, c=1) handled correctly
- ✅ Parameter validation prevents invalid inputs

### Performance Testing
- ✅ Regularized methods achieve same accuracy as unregularized
- ✅ Often improved convergence speed and consistency
- ✅ Robust across different graph types and sizes
- ✅ Proper integration with existing solver framework

The regularization implementation provides a robust, well-tested enhancement to the Motzkin-Straus framework while maintaining full backward compatibility and offering significant improvements in optimization reliability.