# Maximal Clique Finding using Motzkin-Straus Theorem

## Overview

This document describes the implementation of a maximal clique finder based on the Motzkin-Straus theorem, implemented in `scripts/clique_instance.py`. The approach uses JAX-based projected gradient descent (PGD) optimization to find local maxima of the Motzkin-Straus quadratic program, where each local maximum corresponds to a maximal clique.

## Theoretical Foundation

### Motzkin-Straus Theorem
The Motzkin-Straus theorem establishes a connection between the maximum clique problem and quadratic optimization:

- **Global Maximum**: The global maximum of `f(x) = 0.5 * x^T * A * x` over the simplex corresponds to a maximum clique
- **Local Maxima**: Local maxima correspond to maximal cliques (cliques that cannot be extended)
- **Support Extraction**: The support of solution vectors (indices where x_i > threshold) gives clique vertices

### Key Insight
By using multiple restarts of projected gradient descent, we can explore different regions of the optimization landscape and discover multiple local optima, each potentially corresponding to a different maximal clique.

## Code Design

### Architecture
The implementation follows a modular design with clear separation of concerns:

```
scripts/clique_instance.py
├── Core Functions
│   ├── extract_support()           # Extract non-zero indices from solution
│   ├── verify_maximal_clique()     # Verify clique maximality
│   └── find_maximal_cliques_motzkin_straus()  # Main algorithm
├── Visualization
│   └── plot_clique_instances()     # Graph plotting with clique highlighting
├── Testing Infrastructure
│   ├── create_test_graphs()        # Predefined test cases
│   ├── generate_erdos_renyi_graphs()  # Random graph generation
│   └── analyze_clique_coverage()   # Performance analysis
└── CLI Interface
    └── main()                      # Command-line interface with options
```

### Key Design Decisions

1. **Multiple Restarts**: Use 50 restarts by default to explore different local optima
2. **Support Threshold**: Use 1e-5 threshold for extracting significant solution components
3. **Verification Layer**: Verify that extracted supports form valid maximal cliques
4. **JAX Integration**: Leverage existing JAX PGD oracle from the motzkinstraus package
5. **Comprehensive CLI**: Provide extensive command-line options for different use cases

## Code Structure

### Core Algorithm Flow

```python
def find_maximal_cliques_motzkin_straus(graph, num_restarts=50, ...):
    # 1. Convert graph to polynomial format
    poly_indices, poly_coefficients = adjacency_to_polynomial(adj_matrix)
    
    # 2. Generate multiple Dirichlet initializations
    initial_states = sample_dirichlet(key, alpha, (num_restarts,))
    
    # 3. Run PGD optimization from each initialization
    for i in range(num_restarts):
        final_x, _ = run_projected_gradient_descent(...)
        all_final_solutions.append(final_x)
    
    # 4. Extract and verify cliques from each solution
    for solution_vector in all_final_solutions:
        support = extract_support(solution_vector, threshold)
        candidate_clique = map_indices_to_nodes(support)
        
        if verify_clique(graph, candidate_clique) and \
           verify_maximal_clique(graph, candidate_clique):
            maximal_cliques.add(candidate_clique)
    
    return unique_maximal_cliques
```

## Superposition Refinement and Local Search Methods

### Overview
When oracle solutions represent superposition states (multiple cliques encoded in a single solution vector), the algorithm employs sophisticated refinement strategies to extract individual maximal cliques. This is particularly important for quantum annealing solvers like Dirac-3, which may produce solutions that are not pure clique states.

### Greedy Clique Peeling Algorithm

The core refinement method is "Greedy Clique Peeling", which iteratively extracts cliques from superposition solutions:

```python
def refine_superposition_solution(graph, solution_vector, support_indices):
    """
    Extract multiple cliques from a superposition solution using greedy peeling.
    
    Algorithm:
    1. Sort support vertices by solution values (highest first)
    2. For each vertex v in sorted order:
        a. Find all neighbors of v within current support
        b. Form candidate clique with v and its supported neighbors
        c. Verify clique property and extract if valid
        d. Remove extracted vertices from support
        e. Continue until support is exhausted
    
    This approach leverages the insight that higher solution values often
    correspond to vertices that participate in larger or more stable cliques.
    """
```

### Refinement Control Options

The system provides fine-grained control over superposition refinement:

#### CLI Flags
- **`--enable-refinement`** (default): Enable superposition solution refinement
- **`--disable-refinement`**: Skip refinement to improve performance on pure solutions

#### Performance Considerations
- **Refinement Enabled**: More comprehensive clique discovery, higher computational cost
- **Refinement Disabled**: Faster execution, may miss cliques in superposition solutions
- **Automatic Skip**: Refinement is automatically skipped for trivial supports (single vertex)

### Oracle-Specific Refinement Behavior

#### JAX-PGD Oracle
- Pure solutions are common due to local optimization nature
- Refinement helps when gradient descent converges to superposition states
- Multiple restarts naturally provide solution diversity

#### Dirac-3 Oracle  
- Superposition solutions are more common due to quantum nature
- Refinement is crucial for extracting multiple cliques from quantum superposition
- Energy-based solution ranking helps prioritize better refinement candidates

### Refinement Integration

The refinement process is seamlessly integrated into both oracle adapters:

```python
# In oracle adapter find_maximal_cliques() method
if verify_clique(graph, candidate_clique):
    # Pure solution - use directly
    if verify_maximal_clique(graph, candidate_clique):
        maximal_cliques.add(candidate_clique)
else:
    # Superposition solution - attempt refinement if enabled
    if self.enable_refinement and len(support_indices) > 1:
        refined_cliques = self.refine_superposition_solution(
            graph, solution_vector, support_indices
        )
        for refined_clique in refined_cliques:
            if verify_maximal_clique(graph, refined_clique):
                maximal_cliques.add(refined_clique)
    else:
        # Skip refinement - log reason (disabled vs trivial)
        refinement_status = "disabled" if not self.enable_refinement else "trivial support"
        print(f"Skipping refinement ({refinement_status})")
```

### Support Functions

- **`extract_support()`**: Finds indices where solution values exceed threshold
- **`verify_maximal_clique()`**: Checks if clique cannot be extended by adding vertices
- **`refine_superposition_solution()`**: Applies "Greedy Clique Peeling" to extract multiple cliques from superposition solutions
- **`plot_clique_instances()`**: Creates network visualizations with colored clique highlighting
- **`plot_oracle_comparison_results()`**: Enhanced plotting with oracle-specific color coding (blue=JAX-PGD, red=Dirac)
- **`analyze_clique_coverage()`**: Compares found cliques against NetworkX ground truth

### CLI Interface
Comprehensive command-line interface supporting:
- Predefined graph testing (`--test`)
- Erdős-Rényi random graph testing (`--erdos-test`)
- Multiple oracle solvers (`--oracle jax-pgd/dirac`)
- Oracle comparison mode (`--compare-oracles`)
- Superposition refinement control (`--enable-refinement`/`--disable-refinement`)
- Visualization generation (`--plot`)
- Performance comparison (`--compare-networkx`)
- Customizable parameters (restarts, thresholds, etc.)

## Test Results

### Performance on Structured Graphs

| Graph Type | Nodes | Edges | Success Rate | Runtime | Comments |
|------------|-------|-------|--------------|---------|----------|
| Triangle | 3 | 3 | 100% | 11.8s | Perfect match |
| Complete K4 | 4 | 6 | 100% | 12.7s | Perfect match |
| Two Triangles | 6 | 6 | 100% | 9.6s | Found both cliques |
| Diamond | 4 | 5 | 100% | 9.6s | Found both maximal triangles |
| Path P4 | 4 | 3 | 67% | 6.8s | Missing middle edge [1,2] |

**Overall Success Rate on Structured Graphs: 93.3%**

### Performance on Random Graphs (Erdős-Rényi)

| Graph Size | Edge Probability | Restarts | Found Cliques | NetworkX Cliques | Success Rate | Runtime |
|------------|------------------|----------|---------------|------------------|--------------|---------|
| 10 nodes | 0.5 | 50 | 2 | 8 | 25.0% | 24.9s |
| 10 nodes | 0.7 | 100 | 1 | 10 | 10.0% | 66.8s |
| 20 nodes | 0.5 | 50 | 4 | 61 | 6.6% | 83.2s |

### Key Observations

1. **Excellent Performance on Structured Graphs**: Nearly perfect success rates on graphs with clear clique structures
2. **Challenges with Random Graphs**: Lower success rates on Erdős-Rényi graphs due to complex optimization landscape
3. **Scalability**: Runtime increases significantly with graph size and number of restarts
4. **Symmetric Cases**: Some difficulty with symmetric structures (e.g., Path P4 missing middle edge)

## Strengths and Limitations

### Strengths
- **Theoretically Grounded**: Based on solid mathematical foundation (Motzkin-Straus theorem)
- **Multiple Clique Discovery**: Can find multiple maximal cliques in single run
- **Comprehensive Verification**: Validates both clique property and maximality
- **Rich Visualization**: Generates informative network plots with clique highlighting
- **Flexible CLI**: Extensive command-line options for various use cases

### Limitations
- **Incomplete Coverage**: May miss some maximal cliques, especially in dense random graphs
- **Computational Cost**: Requires multiple optimization restarts, leading to higher runtime
- **Parameter Sensitivity**: Performance depends on threshold values and number of restarts
- **Local Optima Challenges**: Optimization landscape complexity can trap algorithm in suboptimal regions

## Usage Examples

### Basic Testing
```bash
# Test on predefined graphs with visualization
python scripts/clique_instance.py --test --compare-networkx --plot

# Test on Erdős-Rényi graphs
python scripts/clique_instance.py --erdos-test --nodes 10 --compare-networkx --plot

# Custom parameters
python scripts/clique_instance.py --test --num-restarts 100 --threshold 1e-6 --verbose
```

### Oracle Comparison and Refinement Control
```bash
# Compare different oracles with color-coded visualization
python scripts/clique_instance.py --test --compare-oracles --plot

# Test with refinement disabled for performance
python scripts/clique_instance.py --test --oracle dirac --disable-refinement

# Enable refinement explicitly (default behavior)
python scripts/clique_instance.py --test --oracle dirac --enable-refinement --verbose

# Compare oracles on Erdős-Rényi graphs
python scripts/clique_instance.py --erdos-test --nodes 10 --compare-oracles --plot
```

### Advanced Usage
```bash
# Performance testing on larger graphs
python scripts/clique_instance.py --erdos-test --nodes 20 --compare-networkx

# Save plots to custom directory
python scripts/clique_instance.py --test --plot --save-plots ./my_plots/

# Dirac oracle with custom configuration and disabled refinement  
python scripts/clique_instance.py --test --oracle dirac --num-samples 200 --disable-refinement
```

## Future Improvements

1. **Adaptive Restart Strategy**: Dynamically adjust number of restarts based on graph properties
2. **Hybrid Approaches**: Combine with other clique-finding algorithms for better coverage
3. **Parameter Optimization**: Automatically tune threshold and restart parameters
4. **Parallel Processing**: Leverage multiple CPU cores for independent restarts
5. **Advanced Initialization**: Use graph-aware initialization strategies beyond Dirichlet sampling

## Conclusion

The Motzkin-Straus based maximal clique finder provides a novel approach to clique detection with strong theoretical foundations. While it excels on structured graphs with clear clique patterns, it faces challenges on complex random graphs where the optimization landscape is more difficult to navigate. The implementation demonstrates the practical applicability of continuous optimization methods to discrete graph problems, offering insights into the connection between local optima and maximal cliques.