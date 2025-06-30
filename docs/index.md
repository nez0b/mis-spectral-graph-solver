# Motzkin-Straus MIS Solver

<div class="grid cards" markdown>

-   :material-graph:{ .lg .middle } **Maximum Independent Set Solver**

    ---

    A comprehensive toolkit for solving the Maximum Independent Set (MIS) problem using the Motzkin-Straus theorem and quantum computing.

-   :material-formula:{ .lg .middle } **Mathematical Foundation**

    ---

    Bridge discrete graph problems to continuous optimization through the elegant Motzkin-Straus theorem:

    $$\max_{x \in \Delta_n} \frac{1}{2} x^T A x = \frac{1}{2}\left(1 - \frac{1}{\omega(G)}\right)$$

-   :material-quantum:{ .lg .middle } **Quantum Computing**

    ---

    Leverage Dirac-3 quantum annealing with advanced parameter control including photon number and quantum fluctuation tuning.

-   :material-speedometer:{ .lg .middle } **High Performance**

    ---

    Multiple solver backends: JAX optimization, Gurobi, SciPy MILP, and hybrid approaches for optimal performance across graph sizes.

</div>

## Overview

The **Motzkin-Straus MIS Solver** transforms the NP-hard Maximum Independent Set problem into a continuous quadratic programming problem using the celebrated Motzkin-Straus theorem. This mathematical bridge enables us to solve discrete graph problems using powerful continuous optimization techniques.

### Key Features

- **🎯 Exact Solutions**: Find optimal maximum independent sets through mathematical optimization
- **⚡ Multiple Solvers**: JAX PGD/Mirror Descent, Gurobi, Dirac-3 quantum annealing, and hybrid approaches  
- **🧮 Mathematical Rigor**: Built on the proven Motzkin-Straus theorem with complete theoretical foundation
- **📊 Comprehensive Benchmarking**: Compare algorithms across performance, quality, and scalability metrics
- **🔬 Research-Ready**: Extensive visualization, analysis tools, and configurable parameters

### Mathematical Foundation

Given a graph $G = (V, E)$ with adjacency matrix $A$, the Motzkin-Straus theorem establishes:

<div class="theorem">
<div class="theorem-title">Motzkin-Straus Theorem</div>

For any graph $G$ with clique number $\omega(G)$, the following equality holds:

$$\max_{x \in \Delta_n} \frac{1}{2} x^T A x = \frac{1}{2}\left(1 - \frac{1}{\omega(G)}\right)$$

where $\Delta_n = \{x \in \mathbb{R}^n | \sum_{i=1}^n x_i = 1, x_i \geq 0\}$ is the probability simplex.
</div>

This theorem allows us to compute the clique number $\omega(G)$ by solving a continuous optimization problem. Since the maximum independent set size equals the maximum clique size in the complement graph, we have $\alpha(G) = \omega(\overline{G})$.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/MotzkinStraus.git
cd MotzkinStraus

# Install with uv (recommended)
uv sync

# Install documentation dependencies (optional)
uv sync --group docs
```

### Basic Usage

```python
import networkx as nx
from motzkinstraus.algorithms import find_mis_with_oracle
from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle

# Create a test graph
G = nx.cycle_graph(8)

# Initialize oracle
oracle = ProjectedGradientDescentOracle(
    learning_rate=0.02,
    max_iterations=1000,
    num_restarts=5
)

# Find maximum independent set
mis_set, oracle_calls = find_mis_with_oracle(G, oracle)
print(f"Maximum Independent Set: {mis_set}")
print(f"Size: {len(mis_set)}, Oracle calls: {oracle_calls}")
```

### Quantum Computing with Dirac-3

```python
from motzkinstraus.oracles.dirac import DiracOracle

# Quantum annealing with advanced parameter control
dirac_oracle = DiracOracle(
    num_samples=50,
    relax_schedule=3,
    sum_constraint=1,
    mean_photon_number=0.002,          # New parameter!
    quantum_fluctuation_coefficient=50  # New parameter!
)

# Use in MIS algorithm
mis_set, calls = find_mis_with_oracle(G, dirac_oracle)
```

## Available Solvers

| Solver | Type | Best For | Complexity |
|--------|------|----------|------------|
| **JAX PGD** | Gradient-based | General purpose | <span class="complexity quadratic">O(iterations × n²)</span> |
| **JAX Mirror Descent** | Entropy-based | Simplex constraints | <span class="complexity quadratic">O(iterations × n²)</span> |
| **Dirac-3** | Quantum annealing | Large problems | <span class="complexity polynomial">O(quantum time)</span> |
| **Gurobi** | Commercial QP | High precision | <span class="complexity polynomial">O(n³)</span> |
| **Hybrid** | Multi-method | Adaptive | <span class="complexity polynomial">Depends on graph</span> |

## Performance Examples

Recent benchmarks on various graph types:

<div class="benchmark-result">
<span class="benchmark-name">15-node Barabási-Albert</span>
<span class="benchmark-time">~0.0004s</span>
<span class="benchmark-quality">100% optimal</span>
</div>

<div class="benchmark-result">
<span class="benchmark-name">JAX PGD (multi-restart)</span>
<span class="benchmark-time">~33s</span>
<span class="benchmark-quality">100% optimal</span>
</div>

<div class="benchmark-result">
<span class="benchmark-name">Dirac-3 Quantum</span>
<span class="benchmark-time">~15s</span>
<span class="benchmark-quality">95%+ optimal</span>
</div>

## What's New in v0.1.0

!!! success "New Dirac-3 API Parameters"

    Enhanced quantum computing capabilities with fine-grained control:
    
    - `mean_photon_number`: Control quantum coherence (range: 6.67×10⁻⁵ to 6.67×10⁻³)
    - `quantum_fluctuation_coefficient`: Tune quantum noise levels (range: 1-100)
    - Complete parameter documentation with physics background

!!! info "Hybrid Solver Framework"

    New hybrid oracles that automatically select the best approach:
    
    - **DiracNetworkXHybridOracle**: Switches between exact and quantum methods
    - **DiracPGDHybridOracle**: Combines quantum global search with local refinement

## Documentation Structure

<div class="grid cards" markdown>

-   **[Theory](theory/motzkin-straus.md)**
    
    Mathematical foundations, theorem proofs, and algorithmic complexity analysis

-   **[API Reference](api/oracles/overview.md)**
    
    Complete documentation for all solvers, oracles, and hybrid methods

-   **[Guides](guides/dirac-configuration.md)**
    
    Practical tutorials for configuration, benchmarking, and performance tuning

-   **[Examples](examples/basic-usage.md)**
    
    Real-world usage scenarios, from basic graphs to large-scale problems

</div>

## Citation

If you use this software in your research, please cite:

```bibtex
@software{motzkinstraus2024,
  title={Motzkin-Straus MIS Solver: Quantum-Enhanced Maximum Independent Set Solutions},
  author={MIS Research Team},
  year={2024},
  url={https://github.com/your-org/MotzkinStraus}
}
```

## Getting Help

- 📖 **Documentation**: Browse the comprehensive guides and API reference
- 🐛 **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/your-org/MotzkinStraus/issues)
- 💬 **Discussions**: Join conversations on [GitHub Discussions](https://github.com/your-org/MotzkinStraus/discussions)
- 📧 **Contact**: Reach out to the research team

---

*Ready to solve maximum independent set problems with mathematical elegance and quantum power? [Get started now!](getting-started/installation.md)*