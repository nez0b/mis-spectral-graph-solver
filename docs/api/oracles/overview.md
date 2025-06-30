# Oracle API Overview

Oracles are the computational engines that solve the Motzkin-Straus quadratic programming problem. Each oracle implements the abstract `Oracle` base class and provides a specialized approach to finding the optimal value of:

$$\max_{x \in \Delta_n} \frac{1}{2} x^T A x$$

## Oracle Architecture

### Base Oracle Class

All oracles inherit from the abstract `Oracle` base class:

```python
from abc import ABC, abstractmethod
import networkx as nx

class Oracle(ABC):
    @abstractmethod
    def solve_quadratic_program(self, adjacency_matrix: np.ndarray) -> float:
        """Solve the Motzkin-Straus QP and return optimal value."""
        pass
    
    def get_omega(self, graph: nx.Graph) -> int:
        """Convert QP solution to clique number using M-S formula."""
        # Implementation handles floating-point robustness
        pass
```

### Key Design Principles

1. **Separation of Concerns**: QP solving vs. discrete conversion
2. **Robustness**: Handle floating-point precision issues
3. **Monitoring**: Track oracle calls and convergence
4. **Flexibility**: Support various solver backends

## Oracle Types

### Classical Optimization

<div class="oracle-card">

#### Gradient-Based Oracles
- **JAX PGD**: Projected gradient descent with simplex projection
- **JAX Mirror Descent**: Exponentiated gradient updates 
- **JAX Frank-Wolfe**: Projection-free conditional gradient method

**Strengths**: Fast, predictable, well-understood convergence  
**Limitations**: May get trapped in local minima

</div>

<div class="oracle-card">

#### Commercial Solvers
- **Gurobi**: Professional-grade non-convex QP solver

**Strengths**: Global optimality guarantees, robust implementation  
**Limitations**: Requires license, can be slow on large problems

</div>

### Quantum Computing

<div class="oracle-card">

#### Quantum Annealing
- **Dirac-3**: Photonic quantum computing with time-bin encoding

**Strengths**: Natural global optimization, handles large problems  
**Limitations**: Quantum hardware access required, approximate solutions

</div>

### Hybrid Approaches

<div class="oracle-card">

#### Adaptive Oracles
- **DiracNetworkXHybrid**: Exact algorithms for small graphs, quantum for large
- **DiracPGDHybrid**: Classical optimization with quantum refinement

**Strengths**: Best of both worlds, adaptive performance  
**Limitations**: Added complexity in configuration

</div>

## Oracle Selection Guide

### By Problem Size

| Graph Size | Recommended Oracle | Rationale |
|------------|-------------------|-----------|
| **< 15 nodes** | GurobiOracle | Exact solutions fast |
| **15-50 nodes** | JAX PGD (multi-restart) | Good balance |
| **50-200 nodes** | DiracOracle | Quantum advantage |
| **> 200 nodes** | Hybrid approaches | Adaptive strategy |

### By Graph Properties

| Graph Type | Best Oracle | Second Choice |
|------------|-------------|---------------|
| **Dense** | JAX Mirror Descent | DiracOracle |
| **Sparse** | JAX PGD | GurobiOracle |
| **Regular** | GurobiOracle | JAX PGD |
| **Scale-free** | DiracOracle | JAX PGD |

### By Requirements

| Requirement | Oracle Choice | Configuration |
|-------------|---------------|---------------|
| **Exact solutions** | GurobiOracle | Default settings |
| **Speed** | JAX PGD | Single restart |
| **Quality** | JAX PGD | Many restarts |
| **Large scale** | DiracOracle | High samples |
| **Production** | Hybrid | Adaptive thresholds |

## Common Usage Patterns

### Basic Oracle Usage

```python
import networkx as nx
from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle

# Create graph
G = nx.karate_club_graph()

# Initialize oracle
oracle = ProjectedGradientDescentOracle()

# Get clique number
omega = oracle.get_omega(G)
print(f"Clique number: {omega}")
print(f"Oracle calls made: {oracle.call_count}")
```

### Multi-Oracle Comparison

```python
from motzkinstraus.oracles import *

oracles = [
    ProjectedGradientDescentOracle(num_restarts=5),
    MirrorDescentOracle(num_restarts=5),
    DiracOracle(num_samples=20)
]

results = {}
for oracle in oracles:
    omega = oracle.get_omega(G)
    results[oracle.name] = {
        'omega': omega,
        'calls': oracle.call_count
    }

print(results)
```

### Performance Monitoring

```python
# Enable verbose mode for detailed tracking
oracle = ProjectedGradientDescentOracle(verbose=True)
oracle.verbose_oracle_calls = True

# Track convergence
omega = oracle.get_omega(G)

# Access detailed metrics
print(f"Final energy: {oracle.last_energy}")
print(f"Iterations: {oracle.last_iterations}")
print(f"Convergence history: {oracle.convergence_history}")
```

### Error Handling

```python
from motzkinstraus.exceptions import OracleError, SolverUnavailableError

try:
    oracle = DiracOracle()  # May fail if Dirac not available
    omega = oracle.get_omega(G)
except SolverUnavailableError as e:
    print(f"Solver not available: {e}")
    # Fallback to available solver
    oracle = ProjectedGradientDescentOracle()
    omega = oracle.get_omega(G)
except OracleError as e:
    print(f"Optimization failed: {e}")
```

## Advanced Configuration

### Automatic Oracle Selection

```python
def get_best_oracle(graph, prefer_exact=True, prefer_quantum=False):
    """Select optimal oracle based on graph properties."""
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    density = 2 * m / (n * (n - 1)) if n > 1 else 0
    
    if n < 15 and prefer_exact:
        return GurobiOracle()
    elif density > 0.7:  # Dense graph
        return MirrorDescentOracle(num_restarts=10)
    elif n > 100 and prefer_quantum:
        return DiracOracle(num_samples=50)
    else:
        return ProjectedGradientDescentOracle(num_restarts=8)

# Usage
oracle = get_best_oracle(G, prefer_exact=False, prefer_quantum=True)
```

### Parameter Optimization

```python
from sklearn.model_selection import ParameterGrid

# Grid search for JAX PGD parameters
param_grid = {
    'learning_rate': [0.01, 0.02, 0.05],
    'num_restarts': [5, 10, 20],
    'tolerance': [1e-6, 1e-7, 1e-8]
}

best_oracle = None
best_score = -float('inf')

for params in ParameterGrid(param_grid):
    oracle = ProjectedGradientDescentOracle(**params)
    # Evaluate on validation graphs
    score = evaluate_oracle(oracle, validation_graphs)
    if score > best_score:
        best_score = score
        best_oracle = oracle
```

## Oracle Metrics and Benchmarking

### Standard Metrics

- **Solution Quality**: How close to optimal
- **Convergence Rate**: Iterations to convergence  
- **Robustness**: Success rate across graph types
- **Scalability**: Performance vs. problem size

### Benchmarking Framework

```python
from motzkinstraus.benchmarks import OracleBenchmark

benchmark = OracleBenchmark(
    oracles=[
        ProjectedGradientDescentOracle(),
        MirrorDescentOracle(),
        DiracOracle()
    ],
    graph_generators=[
        nx.erdos_renyi_graph,
        nx.barabasi_albert_graph,
        nx.watts_strogatz_graph
    ]
)

results = benchmark.run(sizes=[10, 20, 50], repetitions=5)
benchmark.plot_results(results)
```

## Extending the Oracle Framework

### Custom Oracle Implementation

```python
from motzkinstraus.oracles.base import Oracle

class MyCustomOracle(Oracle):
    def __init__(self, custom_param=1.0):
        super().__init__()
        self.custom_param = custom_param
    
    @property
    def name(self):
        return "MyCustom"
    
    @property
    def is_available(self):
        return True  # Check dependencies
    
    def solve_quadratic_program(self, adjacency_matrix):
        # Implement your optimization algorithm
        # Must return optimal value of 0.5 * x.T @ A @ x
        pass
```

### Integration with External Solvers

```python
class ExternalSolverOracle(Oracle):
    def __init__(self, solver_url):
        super().__init__()
        self.solver_url = solver_url
    
    def solve_quadratic_program(self, adjacency_matrix):
        # Send problem to external service
        response = requests.post(self.solver_url, 
                               json={'matrix': adjacency_matrix.tolist()})
        return response.json()['optimal_value']
```

## Oracle Composition Patterns

### Sequential Oracles

```python
class SequentialOracle(Oracle):
    def __init__(self, oracles):
        super().__init__()
        self.oracles = oracles
    
    def solve_quadratic_program(self, adjacency_matrix):
        best_value = -float('inf')
        for oracle in self.oracles:
            try:
                value = oracle.solve_quadratic_program(adjacency_matrix)
                best_value = max(best_value, value)
            except OracleError:
                continue
        return best_value
```

### Ensemble Oracles

```python
class EnsembleOracle(Oracle):
    def __init__(self, oracles, weights=None):
        super().__init__()
        self.oracles = oracles
        self.weights = weights or [1.0] * len(oracles)
    
    def solve_quadratic_program(self, adjacency_matrix):
        values = []
        for oracle in self.oracles:
            value = oracle.solve_quadratic_program(adjacency_matrix)
            values.append(value)
        
        # Weighted average
        return sum(w * v for w, v in zip(self.weights, values)) / sum(self.weights)
```

---

**Next**: Explore individual oracle documentation:
- [JAX Solvers](jax-solvers.md) - Gradient-based optimization methods
- [Dirac-3 Oracle](dirac.md) - Quantum computing approach  
- [Gurobi Oracle](gurobi.md) - Commercial solver integration
- [Hybrid Oracles](hybrid.md) - Combined approaches