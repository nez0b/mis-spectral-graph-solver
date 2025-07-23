# Solution Refinement Methods for Oracle Samples

## Overview

The solution refinement system enables extraction of valid discrete solutions (cliques/independent sets) from continuous optimization results provided by quantum annealing oracles (Dirac-3) and classical optimizers (JAX-PGD). Since these oracles often produce solutions that are energetically favorable but not necessarily valid discrete structures, refinement methods bridge the gap between continuous and discrete optimization.

## Core Architecture

### Universal Refinement Function

The main entry point is `refine_solution_vector()` in `scripts/clique_instance.py` (lines 179-265):

```python
def refine_solution_vector(
    solution_vector: np.ndarray,
    graph: nx.Graph,
    weights: Optional[Dict[int, float]] = None,
    threshold: float = 1e-5,
    max_exact_size: int = 20,
    debug: bool = False
) -> List[Set[int]]:
```

**Key Features:**
- **Problem Type Detection**: Automatically determines if the problem is a maximum clique or maximum weight independent set based on graph structure
- **Multiple Strategies**: Implements three complementary refinement approaches
- **Weighted Support**: Handles both uniform and weighted vertex scenarios
- **Configurable Parameters**: Adjustable thresholds and computational limits
- **Comprehensive Logging**: Detailed debug output for analysis

## Refinement Strategies

### 1. Exact Enumeration Strategy

**Implementation**: `_refine_exact_enumeration()` (lines 268-317)

**Algorithm**:
1. Extract support vertices above threshold
2. Generate all possible subsets up to `max_exact_size`
3. For each subset, verify validity (clique/independent set)
4. Select subset with maximum weight
5. Extend to maximal solution if possible

**Characteristics**:
- **Optimality**: Guarantees optimal solution within size limit
- **Computational Complexity**: O(2^n) where n = min(support_size, max_exact_size)
- **Best For**: Small support sets (≤ 20 vertices) requiring optimal solutions

**Implementation Details**:
```python
# Generate all possible subsets
for r in range(1, min(len(support_list), max_exact_size) + 1):
    for subset in itertools.combinations(support_list, r):
        candidate_vertices = {node_list[idx] for idx in subset}
        
        # Verify validity and track best solution
        if self._is_valid_solution(candidate_vertices, graph, problem_type):
            candidate_weight = sum(vertex_weights.get(v, 1.0) for v in candidate_vertices)
            if candidate_weight > best_weight:
                best_solution = candidate_vertices
                best_weight = candidate_weight
```

### 2. Greedy Peeling Strategy

**Implementation**: `_refine_greedy_peeling()` (lines 320-380)

**Algorithm**:
1. Start with full support set
2. While solution is invalid:
   - Identify vertex causing invalidity (lowest weight among problematic vertices)
   - Remove vertex from candidate set
   - Re-verify validity
3. Extend result to maximal solution

**Characteristics**:
- **Speed**: O(n²) complexity in worst case
- **Heuristic**: May not find optimal solution but guarantees valid result
- **Best For**: Large support sets where exact enumeration is impractical

**Implementation Details**:
```python
current_candidates = support_vertices.copy()

while current_candidates and not self._is_valid_solution(current_candidates, graph, problem_type):
    # Find vertex with minimum weight among current candidates
    min_weight_vertex = min(current_candidates, 
                           key=lambda v: vertex_weights.get(v, 1.0))
    current_candidates.remove(min_weight_vertex)
    
    if debug:
        print(f"    Removed vertex {min_weight_vertex} "
              f"(weight: {vertex_weights.get(min_weight_vertex, 1.0):.4f})")
```

### 3. Greedy Building Strategy

**Implementation**: `_refine_greedy_building()` (lines 383-433)

**Algorithm**:
1. Start with empty solution
2. While vertices remain in support:
   - Select highest-weight vertex compatible with current solution
   - Add to solution if it maintains validity
   - Continue until no more vertices can be added
3. Result is already maximal by construction

**Characteristics**:
- **Incremental**: Builds solution step by step
- **Greedy**: Prioritizes high-weight vertices
- **Natural Maximality**: Produces maximal solutions without additional extension

**Implementation Details**:
```python
remaining_vertices = support_vertices.copy()
current_solution = set()

while remaining_vertices:
    # Find highest-weight compatible vertex
    compatible_vertices = []
    for vertex in remaining_vertices:
        test_solution = current_solution | {vertex}
        if self._is_valid_solution(test_solution, graph, problem_type):
            compatible_vertices.append(vertex)
    
    if not compatible_vertices:
        break
    
    # Add highest-weight compatible vertex
    next_vertex = max(compatible_vertices, 
                     key=lambda v: vertex_weights.get(v, 1.0))
    current_solution.add(next_vertex)
    remaining_vertices.remove(next_vertex)
```

## Problem Type Support

### Maximum Clique Problems

**Validation**: Direct adjacency checking
```python
def _is_clique(vertices: Set[int], graph: nx.Graph) -> bool:
    for u in vertices:
        for v in vertices:
            if u != v and not graph.has_edge(u, v):
                return False
    return True
```

### Maximum Weight Independent Set Problems

**Validation**: Convert to clique problem using graph complement
```python
def _is_independent_set(vertices: Set[int], graph: nx.Graph) -> bool:
    # Convert to clique problem on complement graph
    complement_graph = nx.complement(graph)
    return self._is_clique(vertices, complement_graph)
```

**Automatic Detection**: Based on graph density and structure patterns

## Maximal Extension

**Implementation**: `_extend_to_maximal()` (lines 436-492)

Extends any valid solution to a maximal one by greedily adding vertices:

```python
def _extend_to_maximal(
    vertices: Set[int], 
    graph: nx.Graph, 
    problem_type: str,
    vertex_weights: Dict[int, float]
) -> Set[int]:
    extended = vertices.copy()
    
    while True:
        # Find vertices that can be added while maintaining validity
        candidates = []
        for vertex in graph.nodes():
            if vertex not in extended:
                test_set = extended | {vertex}
                if self._is_valid_solution(test_set, graph, problem_type):
                    candidates.append(vertex)
        
        if not candidates:
            break  # No more vertices can be added - maximal reached
        
        # Add highest-weight candidate
        best_candidate = max(candidates, key=lambda v: vertex_weights.get(v, 1.0))
        extended.add(best_candidate)
    
    return extended
```

## Oracle Integration

### Dirac-3 Quantum Annealing Adapter

**Location**: `scripts/oracles/dirac_adapter.py` (lines 705-750)

**Integration**:
```python
# Replace simple validity check with comprehensive refinement
refined_cliques = refine_solution_vector(
    solution_vector=solution_vector,
    graph=graph,
    weights=weights,  # Uses quantum annealing weights
    threshold=support_threshold,
    max_exact_size=20,
    debug=self.verbose
)

# Process all refined solutions
for refined_clique in refined_cliques:
    if self.verify_maximal_clique(graph, refined_clique):
        clique_frozen = frozenset(refined_clique)
        if clique_frozen not in maximal_cliques:
            maximal_cliques.add(clique_frozen)
```

### JAX-PGD Projected Gradient Descent Adapter

**Location**: `scripts/oracles/jax_pgd_adapter.py` (lines 245-272)

**Integration**:
```python
# Use unified refinement for invalid solutions
refined_cliques = refine_solution_vector(
    solution_vector=solution_vector,
    graph=graph,
    weights=None,  # JAX-PGD uses uniform weights for clique problems
    threshold=support_threshold,
    max_exact_size=20,
    debug=self.verbose
)
```

## Performance Characteristics

### Computational Complexity

| Strategy | Time Complexity | Space Complexity | Optimality | Best Use Case |
|----------|----------------|------------------|------------|---------------|
| Exact Enumeration | O(2^min(n,k)) | O(n) | Optimal | Small supports (≤20) |
| Greedy Peeling | O(n²) | O(n) | Heuristic | Large supports |
| Greedy Building | O(n²) | O(n) | Heuristic | Incremental construction |

### Strategy Selection Logic

```python
# Automatic strategy selection based on support size
if len(support_vertices) <= max_exact_size:
    # Use exact enumeration for guaranteed optimality
    return self._refine_exact_enumeration(...)
elif problem_requires_high_quality:
    # Use greedy building for better solution quality
    return self._refine_greedy_building(...)
else:
    # Use greedy peeling for speed
    return self._refine_greedy_peeling(...)
```

## Debug Output Features

### Comprehensive Logging

When `debug=True`, the system provides detailed output:

```
Refining solution vector with 847 variables
  Solution sum: 23.45, max: 0.89, non-zero entries: 12
  Problem type: maximum_clique (density: 0.31)
  Support vertices: {2, 5, 8, 12, 15, 18, 23, 27, 31, 35, 40, 44}
  
  Strategy: exact_enumeration (support size 12 ≤ max_exact_size 20)
  
  Checking subset {2, 5, 8}: VALID clique, weight: 2.47
  Checking subset {2, 5, 12}: INVALID (no edge 5-12)
  ...
  Best valid subset: {2, 5, 8, 15} with weight: 3.21
  
  Extending to maximal...
  Added vertex 23 (weight: 0.45)
  Added vertex 31 (weight: 0.38)
  Final maximal clique: {2, 5, 8, 15, 23, 31} (size: 6, weight: 4.04)
```

### Performance Metrics

```python
# Timing and statistics tracking
refinement_start = time.time()
# ... refinement process ...
refinement_time = time.time() - refinement_start

if debug:
    print(f"  Refinement completed in {refinement_time:.3f}s")
    print(f"  Evaluated {num_candidates} candidates")
    print(f"  Found {len(results)} valid solutions")
```

## Configuration Parameters

### Key Parameters

- **`threshold`** (default: 1e-5): Minimum value to consider vertex as "active"
- **`max_exact_size`** (default: 20): Maximum support size for exact enumeration
- **`debug`** (default: False): Enable comprehensive logging
- **`weights`** (optional): Custom vertex weights for optimization

### Tuning Guidelines

- **Small Graphs (≤ 50 vertices)**: Use exact enumeration with higher `max_exact_size`
- **Large Graphs (> 100 vertices)**: Rely on greedy strategies, lower `threshold`
- **High-Quality Solutions**: Enable all strategies, use exact enumeration when possible
- **Speed-Critical Applications**: Use greedy peeling only, disable debug output

## Error Handling and Robustness

### Graceful Degradation

```python
try:
    # Attempt exact enumeration
    result = self._refine_exact_enumeration(...)
except (MemoryError, TimeoutError):
    # Fall back to greedy methods
    result = self._refine_greedy_peeling(...)
```

### Input Validation

```python
# Comprehensive input checking
if len(solution_vector) != graph.number_of_nodes():
    raise ValueError("Solution vector length must match graph size")

if threshold < 0:
    raise ValueError("Threshold must be non-negative")

if not graph.nodes():
    return []  # Empty graph
```

## Future Extensions

### Planned Enhancements

1. **Machine Learning Integration**: Use neural networks to predict optimal strategy
2. **Parallel Processing**: Distribute exact enumeration across multiple cores
3. **Advanced Heuristics**: Implement simulated annealing and tabu search
4. **Adaptive Thresholds**: Dynamic threshold adjustment based on solution quality
5. **Graph-Specific Optimizations**: Specialized algorithms for planar, bipartite, and sparse graphs

### Research Directions

1. **Quantum-Classical Hybrid**: Combine quantum annealing with classical refinement
2. **Approximation Guarantees**: Theoretical bounds on solution quality
3. **Online Learning**: Adaptive strategy selection based on problem characteristics
4. **Multi-Objective Optimization**: Balance solution quality vs. computational cost

## Usage Examples

### Basic Usage

```python
from scripts.clique_instance import refine_solution_vector

# Refine oracle solution
refined_cliques = refine_solution_vector(
    solution_vector=oracle_result,
    graph=problem_graph,
    threshold=1e-5,
    debug=True
)

print(f"Found {len(refined_cliques)} valid cliques")
```

### Advanced Configuration

```python
# Custom weights and strategy selection
refined_solutions = refine_solution_vector(
    solution_vector=weighted_solution,
    graph=large_graph,
    weights=custom_vertex_weights,
    threshold=1e-3,  # Higher threshold for noisy solutions
    max_exact_size=15,  # Lower limit for large graphs
    debug=False  # Disable for production
)
```

This refinement system transforms raw oracle outputs into high-quality discrete solutions, significantly improving the practical utility of quantum annealing and classical continuous optimization in combinatorial problems.