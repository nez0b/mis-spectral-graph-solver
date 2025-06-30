# Computing ω(G): Maximum Clique Size Tutorial

This tutorial teaches you how to compute the **maximum clique size** ω(G) using the `motzkinstraus` package. We'll start with simple examples and build up to advanced usage, connecting the theoretical [Motzkin-Straus theorem](../theory/motzkin-straus.md) to practical implementation.

## Mathematical Context

The **clique number** ω(G) is the size of the largest complete subgraph (clique) in graph G:

- **Complete Graph K₅**: ω(K₅) = 5 (all vertices form one clique)
- **Cycle Graph C₅**: ω(C₅) = 2 (any two adjacent vertices)
- **Tree**: ω(Tree) = 2 (any edge forms a clique)
- **Independent Set**: ω(Empty) = 1 (no edges, so single vertices only)

The Motzkin-Straus theorem enables us to compute ω(G) by solving a continuous optimization problem instead of exhaustive search.

## Quick Start

Get ω(G) in just 3 lines of code:

```python
import networkx as nx
from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle

# Create a graph
graph = nx.complete_graph(5)  # K₅ has ω(G) = 5

# Compute clique number
oracle = ProjectedGradientDescentOracle()
omega = oracle.get_omega(graph)
print(f"ω(G) = {omega}")  # Output: ω(G) = 5
```

That's it! The oracle automatically handles the Motzkin-Straus optimization and returns the clique number.

## Which Oracle Should I Use?

The library provides multiple solver backends, called **oracles**. Each has unique strengths for different scenarios:

| Oracle | Solver Type | Accuracy | Speed | Best For |
|--------|-------------|----------|-------|----------|
| **JAX Oracles** | Continuous Optim. | Approximate | Fast (GPU support) | Large graphs, quick estimates, research |
| **MILP Oracles** | Integer Programming | **Exact** | Slow (Exponential) | Small-medium graphs where proof of optimality is required |
| **NetworkX Oracle** | Exact Algorithm | **Exact** | Very Slow (Exponential) | Small graphs, validation, educational purposes |
| **Dirac Oracle** | Quantum Inspired | Approximate | Varies (Hardware dep.) | Experimental use, exploring quantum approaches |

### Quick Decision Guide

- **For most applications**: Start with `ProjectedGradientDescentOracle` (JAX)
- **When you need guaranteed correctness**: Use MILP solvers and be prepared for longer runtimes
- **For learning or validating small graphs**: NetworkX is simplest
- **For research into quantum methods**: Explore the Dirac oracle

## Understanding the Oracles

### JAX Oracles: Fast Continuous Optimization

JAX oracles implement the Motzkin-Straus theorem directly using modern optimization algorithms:

```python
from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle
from motzkinstraus.oracles.jax_mirror import MirrorDescentOracle

# Projected Gradient Descent (recommended for most cases)
pgd_oracle = ProjectedGradientDescentOracle(
    learning_rate=0.02,
    max_iterations=500,
    num_restarts=5,
    tolerance=1e-6
)

# Mirror Descent (alternative optimization approach)
mirror_oracle = MirrorDescentOracle(
    learning_rate=0.01,
    max_iterations=500,
    num_restarts=5
)

# Both work the same way
graph = nx.erdos_renyi_graph(50, 0.3)
omega_pgd = pgd_oracle.get_omega(graph)
omega_mirror = mirror_oracle.get_omega(graph)

print(f"PGD result: ω(G) = {omega_pgd}")
print(f"Mirror Descent result: ω(G) = {omega_mirror}")
```

**How it works**: These oracles solve the optimization problem `max x^T A x` subject to `x ∈ Δₙ` (probability simplex), then use the Motzkin-Straus formula `ω(G) = 1/(1 - 2M)` where M is the optimal value.

### MILP Oracles: Exact Solutions

MILP (Mixed-Integer Linear Programming) oracles provide **provably optimal** results by formulating clique detection as an integer program:

```python
from motzkinstraus.oracles.gurobi import GurobiOracle
from motzkinstraus.solvers.scipy_milp import get_clique_number_scipy

# Gurobi Oracle (requires Gurobi license)
if GurobiOracle.is_available():
    gurobi_oracle = GurobiOracle(suppress_output=True)
    omega_exact = gurobi_oracle.get_omega(graph)
    print(f"Gurobi (exact): ω(G) = {omega_exact}")

# SciPy MILP (free alternative)
omega_scipy = get_clique_number_scipy(graph)
print(f"SciPy MILP (exact): ω(G) = {omega_scipy}")
```

**When to use**: Small to medium graphs (< 100 nodes) where you need mathematical proof of optimality. Runtime grows exponentially with graph size.

### NetworkX Oracle: Simple Baseline

```python
import networkx as nx

# Built-in NetworkX exact solver
omega_nx = nx.graph_clique_number(graph)
print(f"NetworkX (exact): ω(G) = {omega_nx}")
```

**When to use**: Educational purposes, small graphs (< 20 nodes), or as a validation baseline for other methods.

### Dirac Oracle: Quantum-Inspired Optimization

The Dirac oracle explores quantum annealing approaches to optimization:

```python
from motzkinstraus.oracles.dirac import DiracOracle

# Quantum-inspired solver
dirac_oracle = DiracOracle(
    num_samples=50,
    relax_schedule=3,
    mean_photon_number=0.0015,
    quantum_fluctuation_coefficient=3
)

if dirac_oracle.is_available:
    omega_quantum = dirac_oracle.get_omega(graph)
    print(f"Dirac (quantum): ω(G) = {omega_quantum}")
```

**When to use**: Research into quantum computing applications, experimenting with novel optimization paradigms, or when classical methods struggle with specific graph structures.

## Practical Workshop: Comparing Methods

Let's compare different approaches on real examples:

```python
import networkx as nx
import time
from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle
from motzkinstraus.solvers.scipy_milp import get_clique_number_scipy

def compare_methods(graph, graph_name):
    """Compare different ω(G) computation methods."""
    print(f"\n=== {graph_name} ===")
    print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
    
    results = {}
    
    # JAX Oracle (fast, approximate)
    oracle = ProjectedGradientDescentOracle(num_restarts=3, verbose=False)
    start = time.time()
    omega_jax = oracle.get_omega(graph)
    time_jax = time.time() - start
    results['JAX PGD'] = (omega_jax, time_jax, 'Approximate')
    
    # NetworkX (exact, slow)
    if graph.number_of_nodes() <= 15:  # Only for small graphs
        start = time.time()
        omega_nx = nx.graph_clique_number(graph)
        time_nx = time.time() - start
        results['NetworkX'] = (omega_nx, time_nx, 'Exact')
    
    # SciPy MILP (exact, moderate speed)
    if graph.number_of_nodes() <= 25:  # Only for small-medium graphs
        start = time.time()
        omega_milp = get_clique_number_scipy(graph)
        time_milp = time.time() - start
        results['SciPy MILP'] = (omega_milp, time_milp, 'Exact')
    
    # Display results
    print(f"{'Method':<12} {'ω(G)':<6} {'Time(s)':<10} {'Type':<12}")
    print("-" * 40)
    for method, (omega, runtime, type_) in results.items():
        print(f"{method:<12} {omega:<6} {runtime:<10.4f} {type_:<12}")
    
    return results

# Test on different graph types
examples = [
    (nx.complete_graph(8), "Complete K₈"),
    (nx.cycle_graph(10), "Cycle C₁₀"),
    (nx.erdos_renyi_graph(20, 0.4), "Erdős-Rényi(20, 0.4)"),
    (nx.karate_club_graph(), "Karate Club"),
]

for graph, name in examples:
    compare_methods(graph, name)
```

## Advanced Configuration

### Tuning JAX Oracles

The JAX oracles support extensive configuration to optimize performance:

```python
from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle

# High-precision configuration for accuracy
precise_oracle = ProjectedGradientDescentOracle(
    learning_rate=0.01,        # Smaller steps for stability
    max_iterations=2000,       # More iterations for convergence
    num_restarts=10,           # Multiple restarts to avoid local optima
    tolerance=1e-8,            # Tight convergence criterion
    verbose=True               # Show optimization progress
)

# Fast configuration for quick estimates
fast_oracle = ProjectedGradientDescentOracle(
    learning_rate=0.05,        # Larger steps for speed
    max_iterations=200,        # Fewer iterations
    num_restarts=3,            # Fewer restarts
    tolerance=1e-4,            # Looser convergence
    verbose=False
)

# GPU configuration (if available)
gpu_oracle = ProjectedGradientDescentOracle(
    learning_rate=0.02,
    max_iterations=1000,
    num_restarts=8,
    use_gpu=True               # Enable GPU acceleration
)
```

### Configuration Parameters Guide

| Parameter | JAX Oracles | Description |
|-----------|-------------|-------------|
| `learning_rate` | PGD, Mirror | Step size for optimization (0.001-0.1 typical) |
| `max_iterations` | PGD, Mirror | Maximum optimization steps (100-2000 typical) |
| `num_restarts` | PGD, Mirror | Random restarts to avoid local optima (1-20 typical) |
| `tolerance` | PGD, Mirror | Convergence threshold (1e-8 to 1e-4 typical) |
| `verbose` | PGD, Mirror | Show detailed optimization output |

| Parameter | MILP Oracles | Description |
|-----------|--------------|-------------|
| `solver` | Gurobi, SciPy | Backend solver ('GUROBI', 'SCIPY', etc.) |
| `time_limit` | Gurobi | Maximum solve time in seconds |
| `suppress_output` | Gurobi | Hide solver console output |

| Parameter | Dirac Oracle | Description |
|-----------|--------------|-------------|
| `num_samples` | Dirac | Number of quantum annealing samples |
| `relax_schedule` | Dirac | Relaxation schedule parameter |
| `mean_photon_number` | Dirac | Quantum photon number setting |

## Working with DIMACS Files

Many graph theory benchmarks use the DIMACS format. Here's how to compute ω(G) for DIMACS files:

```python
import networkx as nx
from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle

def read_dimacs_graph(filename):
    """Read a graph from DIMACS format."""
    graph = nx.Graph()
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('p edge'):
                # Problem line: p edge <num_vertices> <num_edges>
                parts = line.strip().split()
                num_vertices = int(parts[2])
                graph.add_nodes_from(range(1, num_vertices + 1))
            elif line.startswith('e'):
                # Edge line: e <vertex1> <vertex2>
                parts = line.strip().split()
                u, v = int(parts[1]), int(parts[2])
                graph.add_edge(u, v)
    
    return graph

def process_dimacs_file(filename):
    """Process a DIMACS file and compute ω(G)."""
    print(f"Processing {filename}...")
    
    # Load graph
    graph = read_dimacs_graph(filename)
    print(f"Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Compute ω(G) using JAX oracle
    oracle = ProjectedGradientDescentOracle(num_restarts=5, verbose=False)
    start_time = time.time()
    omega = oracle.get_omega(graph)
    runtime = time.time() - start_time
    
    print(f"Result: ω(G) = {omega} (computed in {runtime:.3f}s)")
    return omega, runtime

# Example usage
# omega, time = process_dimacs_file("path/to/graph.dimacs")
```

## Troubleshooting

### Common Issues and Solutions

**Issue**: JAX oracle returns obviously wrong results  
**Solution**: Increase `num_restarts` and `max_iterations`. The optimization may be stuck in local optima.

```python
# Instead of default settings
oracle = ProjectedGradientDescentOracle()

# Try this for difficult graphs
oracle = ProjectedGradientDescentOracle(
    num_restarts=10,
    max_iterations=1000,
    tolerance=1e-7
)
```

**Issue**: MILP solver takes too long  
**Solution**: Set time limits or switch to approximate methods for large graphs.

```python
from motzkinstraus.oracles.gurobi import GurobiOracle

# Set 60-second time limit
oracle = GurobiOracle(time_limit=60)
```

**Issue**: "Oracle not available" errors  
**Solution**: Check dependencies and installation.

```python
from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle
from motzkinstraus.oracles.gurobi import GurobiOracle

# Check availability before using
pgd_oracle = ProjectedGradientDescentOracle()
if pgd_oracle.is_available:
    print("JAX PGD Oracle available")
else:
    print("JAX PGD Oracle not available - check JAX installation")

if GurobiOracle.is_available():
    print("Gurobi Oracle available")
else:
    print("Gurobi Oracle not available - check license")
```

### Validation and Verification

Always validate approximate results against exact methods when possible:

```python
def validate_omega_result(graph, approximate_omega, tolerance=0):
    """Validate approximate ω(G) against exact methods."""
    if graph.number_of_nodes() <= 15:
        exact_omega = nx.graph_clique_number(graph)
        diff = abs(approximate_omega - exact_omega)
        
        if diff <= tolerance:
            print(f"✅ Validation passed: {approximate_omega} ≈ {exact_omega}")
            return True
        else:
            print(f"⚠️ Validation failed: {approximate_omega} vs {exact_omega} (diff: {diff})")
            return False
    else:
        print(f"⚡ Graph too large for exact validation ({graph.number_of_nodes()} nodes)")
        return None

# Example validation
graph = nx.erdos_renyi_graph(12, 0.4)
oracle = ProjectedGradientDescentOracle(num_restarts=5)
omega_approx = oracle.get_omega(graph)
validate_omega_result(graph, omega_approx)
```

## Summary

You now know how to:

1. **Compute ω(G)** using the unified `get_omega()` interface
2. **Choose the right oracle** based on your accuracy/speed requirements
3. **Configure oracles** for optimal performance on your graphs
4. **Validate results** using multiple methods
5. **Handle real-world data** including DIMACS format files

The `motzkinstraus` package transforms the theoretical elegance of the Motzkin-Straus theorem into practical tools for solving maximum clique problems. Whether you need fast approximations for large-scale analysis or exact solutions for critical applications, there's an oracle designed for your needs.

### Next Steps

- Explore the [Maximum Independent Set tutorial](omega-computation.md) to learn about the dual problem
- Read about [oracle implementation details](../api/oracles/overview.md) 
- Try the [Dirac quantum oracle guide](../guides/dirac-configuration.md) for cutting-edge approaches
- Check out [performance tuning tips](../guides/performance-tuning.md) for large-scale applications