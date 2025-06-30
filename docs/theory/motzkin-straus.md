# The Motzkin-Straus Theorem

The Motzkin-Straus theorem provides an elegant bridge between discrete graph theory and continuous optimization, enabling us to solve the NP-hard Maximum Independent Set problem through quadratic programming.

## Problem Definition

### Maximum Independent Set (MIS)

!!! abstract "Definition: Independent Set"
    
    Given an undirected graph $G = (V, E)$, an **independent set** is a subset $S \subseteq V$ of vertices such that no two vertices in $S$ are adjacent:
    
    $$\forall u, v \in S: (u, v) \notin E$$

The **Maximum Independent Set (MIS) problem** seeks to find an independent set of maximum cardinality:

$$\alpha(G) = \max\{|S| : S \text{ is an independent set in } G\}$$

where $\alpha(G)$ is called the **independence number** of $G$.

### Maximum Clique Problem

!!! abstract "Definition: Clique"
    
    A **clique** in graph $G = (V, E)$ is a subset $C \subseteq V$ where every pair of distinct vertices is adjacent:
    
    $$\forall u, v \in C, u \neq v: (u, v) \in E$$

The **clique number** $\omega(G)$ is the size of the maximum clique:

$$\omega(G) = \max\{|C| : C \text{ is a clique in } G\}$$

### Fundamental Relationship

The key insight connecting these problems lies in the **complement graph**:

!!! info "Graph Complement"
    
    For graph $G = (V, E)$, the complement $\overline{G} = (V, \overline{E})$ has the same vertices but complementary edges:
    
    $$\overline{E} = \{(u, v) : u, v \in V, u \neq v, (u, v) \notin E\}$$

This leads to the fundamental equivalence:

<div class="theorem">
<div class="theorem-title">MIS-Clique Equivalence</div>

For any graph $G$:

$$\alpha(G) = \omega(\overline{G})$$

An independent set in $G$ corresponds exactly to a clique in $\overline{G}$.
</div>

## The Motzkin-Straus Theorem

### Statement

<div class="theorem">
<div class="theorem-title">Motzkin-Straus Theorem (1965)</div>

Let $G = (V, E)$ be a graph with $n = |V|$ vertices and adjacency matrix $A$. Define the **standard simplex**:

$$\Delta_n = \left\{x \in \mathbb{R}^n : \sum_{i=1}^n x_i = 1, x_i \geq 0 \text{ for all } i\right\}$$

Then the following equality holds:

$$\max_{x \in \Delta_n} \frac{1}{2} x^T A x = \frac{1}{2}\left(1 - \frac{1}{\omega(G)}\right)$$

where $\omega(G)$ is the clique number of $G$.
</div>

### Intuition

The theorem establishes that:

1. **Left side**: A continuous quadratic optimization problem over the probability simplex
2. **Right side**: A simple function of the discrete clique number

This remarkable connection allows us to solve discrete graph problems using continuous optimization techniques.

## Algorithmic Derivation from the Original Paper

The 1965 Motzkin-Straus paper not only proves the theorem but reveals how the algorithm emerges naturally from the mathematical analysis. This section follows their original derivation.

### Mathematical Foundation

Given graph $G$ with vertices $1, 2, \ldots, n$, we seek to maximize:

$$F(x) = \sum_{(i,j) \in E(G)} x_i x_j$$

subject to $x \in \Delta_n$ (the probability simplex).

### Proof Strategy and Algorithm Emergence

The original proof uses **mathematical induction** and **critical point analysis**, which naturally leads to an iterative algorithm:

#### Step 1: Lower Bound Construction (Algorithmic Insight)

For any clique $C$ of size $k$ in $G$, setting:
- $x_i = \frac{1}{k}$ for all $i \in C$
- $x_j = 0$ for all $j \notin C$

yields the objective value:

$$F(x) = \binom{k}{2} \cdot \frac{1}{k^2} = \frac{1}{2}\left(1 - \frac{1}{k}\right)$$

**Algorithmic Implication**: The optimal solution concentrates probability mass uniformly on the maximum clique.

#### Step 2: Optimality Conditions (Algorithm Discovery)

At an interior maximum, all partial derivatives must be equal (Lagrange multipliers):

$$\frac{\partial F}{\partial x_i} = \sum_{j: (i,j) \in E(G)} x_j = \lambda \quad \text{for all } i \text{ with } x_i > 0$$

**Key Insight**: This condition reveals the iterative algorithm! Each vertex's "score" in the optimal solution equals its weighted degree in the current solution.

#### Step 3: Inductive Argument (Boundary Analysis)

The original proof analyzes two cases:

**Case A**: Maximum occurs on the boundary of the simplex
- Some $x_i = 0$, reducing to a smaller graph
- Apply induction hypothesis

**Case B**: Maximum occurs in the interior
- All optimality conditions hold simultaneously
- Graph must have special structure (complete subgraph)

**Algorithmic Emergence**: This analysis suggests checking whether including/excluding vertices improves the objective.

### The Natural Algorithm

From the optimality conditions, we derive the **iterative reweighting algorithm**:

<div class="algorithm">
<div class="algorithm-title">Motzkin-Straus Iterative Algorithm</div>

**Input**: Graph $G$ with adjacency matrix $A$  
**Output**: Optimal vector $x^*$ and clique number $\omega(G)$

1. **Initialize**: $x^{(0)} \leftarrow \frac{1}{n} \mathbf{1}$ (uniform distribution)
2. **Iterate** until convergence:
   - **Compute scores**: $s_i^{(t)} = \sum_{j: (i,j) \in E(G)} x_j^{(t)} = (Ax^{(t)})_i$
   - **Reweight**: $x_i^{(t+1)} = \frac{s_i^{(t)}}{\sum_{j=1}^n s_j^{(t)}}$
3. **Extract**: $\omega(G) = \frac{1}{1 - 2F(x^*)}$ where $F(x^*) = \frac{1}{2}(x^*)^T A x^*$
</div>

### Mathematical Justification of the Algorithm

The reweighting rule $x_i^{(t+1)} \propto (Ax^{(t)})_i$ emerges from the optimality condition:

$$\frac{\partial F}{\partial x_i}\bigg|_{x=x^*} = \sum_{j: (i,j) \in E(G)} x_j^* = \text{constant for all } i \text{ with } x_i^* > 0$$

This is precisely the **replicator dynamics** from evolutionary game theory!

### Proof Completion (Original Method)

#### Lemma 1: Boundary Reduction
If the maximum occurs when some $x_i = 0$, then we can delete vertex $i$ and solve the reduced problem.

#### Lemma 2: Interior Structure
If the maximum occurs in the interior, then the graph restricted to $\{i : x_i > 0\}$ must be complete.

**Proof technique**: If $(i,j) \notin E(G)$ but $x_i, x_j > 0$, then the transformation:
$$x_i \leftarrow x_i - \epsilon, \quad x_j \leftarrow x_j + \epsilon$$
preserves the simplex constraint but can increase the objective, contradicting optimality.

#### Main Theorem Proof
Combining the lemmas with induction on $n$:

1. **Base case**: $n = 1$ gives $\omega(G) = 1$ and $F(x) = 0$
2. **Inductive step**: Either apply boundary reduction or use interior structure
3. **Conclusion**: $\max F(x) = \frac{1}{2}(1 - \frac{1}{\omega(G)})$

### Connection to Modern Optimization

The original derivation connects to several modern approaches:

#### Frank-Wolfe Method
The iterative algorithm is a special case of Frank-Wolfe:
- **Linear oracle**: Find $v = \arg\max_{u \in \Delta_n} \langle \nabla F(x^{(t)}), u \rangle$
- **Update**: $x^{(t+1)} = (1-\gamma_t)x^{(t)} + \gamma_t v$

#### Spectral Methods
When $A$ is the adjacency matrix, the algorithm finds the principal eigenvector direction restricted to the simplex.

#### Interior Point Methods
Modern solvers handle the simplex constraints directly using barrier methods or active-set approaches.

### Convergence Analysis

From the original paper's perspective:

!!! note "Convergence Properties"
    
    - **Fixed points**: Correspond to local maxima of $F(x)$ on $\Delta_n$
    - **Global maximum**: Achieved when support of $x^*$ is exactly a maximum clique
    - **Rate**: Depends on spectral gap of the reweighting operator
    - **Initialization**: Different starting points may converge to different local maxima

## Computational Implications

### Oracle Construction

The Motzkin-Straus theorem enables us to construct an **oracle** for the clique number:

<div class="algorithm">
<div class="algorithm-title">Oracle(G) → ω(G)</div>

**Input**: Graph $G$ with adjacency matrix $A$  
**Output**: Clique number $\omega(G)$

1. **Formulate QP**: $\max_{x \in \Delta_n} \frac{1}{2} x^T A x$
2. **Solve**: Use numerical optimization to find optimal value $M$
3. **Convert**: Return $\omega(G) = \frac{1}{1 - 2M}$
</div>

### MIS Algorithm

Combined with the MIS-clique equivalence, we get our complete algorithm:

<div class="algorithm">
<div class="algorithm-title">MIS Algorithm</div>

**Input**: Graph $G$  
**Output**: Maximum independent set of $G$

1. **Target Size**: Compute $k = \alpha(G) = \omega(\overline{G})$ using oracle
2. **Iterative Construction**: Build MIS vertex by vertex using oracle queries
3. **Optimality Check**: For each vertex $v$, test if including $v$ maintains optimality
4. **Decision**: Include $v$ if $1 + \text{remaining\_capacity} = \text{current\_target}$
</div>

## Quadratic Programming Formulation

### Standard Form

The optimization problem can be written in standard quadratic programming form:

$$\begin{align}
\text{maximize} \quad & \frac{1}{2} x^T A x \\
\text{subject to} \quad & \sum_{i=1}^n x_i = 1 \\
& x_i \geq 0, \quad i = 1, \ldots, n
\end{align}$$

### Properties

!!! note "Problem Characteristics"
    
    - **Objective**: Quadratic (generally non-convex for graph adjacency matrices)
    - **Constraints**: Linear (defines the probability simplex)
    - **Variables**: $n$ continuous variables
    - **Feasible Region**: Compact and convex
    - **Global Optimum**: Guaranteed to exist (Weierstrass theorem)

### Challenges

1. **Non-convexity**: The objective function $\frac{1}{2} x^T A x$ is generally non-convex
2. **Local Optima**: Standard optimization algorithms may get trapped
3. **Numerical Precision**: Floating-point errors can affect the final integer result

## Extensions and Variations

### Weighted Graphs

The theorem extends naturally to weighted graphs with adjacency matrix $A$ having non-negative weights:

$$\max_{x \in \Delta_n} \frac{1}{2} x^T A x = \frac{1}{2}\left(1 - \frac{1}{\omega_w(G)}\right)$$

where $\omega_w(G)$ is the weight of the maximum weighted clique.

### Fractional Clique Number

The continuous relaxation provides insights into the **fractional clique number**:

$$\omega_f(G) = \frac{1}{1 - 2M}$$

where $M$ is the optimal value of the quadratic program. This quantity satisfies:

$$\omega(G) \leq \omega_f(G) \leq n$$

### Stability Number Connection

For the **stability number** (independence number), we have:

$$\alpha(G) = \omega(\overline{G}) = \frac{1}{1 - 2 \max_{x \in \Delta_n} \frac{1}{2} x^T \overline{A} x}$$

where $\overline{A}$ is the adjacency matrix of the complement graph.

## Theoretical Complexity

### Classical Complexity

- **MIS Problem**: NP-hard in general graphs
- **Approximation**: Hard to approximate within factor $n^{1-\epsilon}$ for any $\epsilon > 0$
- **Special Cases**: Polynomial-time solvable in bipartite graphs, trees, etc.

### Continuous Relaxation

The quadratic program provides:

- **Continuous Relaxation**: Polynomial-time solvable (in theory)
- **Rounding**: The continuous solution must be carefully rounded to discrete
- **Approximation Quality**: Depends on the optimization algorithm used

### Our Approach Complexity

The oracle-based MIS algorithm has complexity:

- **Oracle Calls**: $O(n)$ calls in the worst case
- **Per-Oracle Cost**: Depends on the quadratic programming solver
- **Total Complexity**: $O(n \cdot T_{QP}(n))$ where $T_{QP}(n)$ is QP solver time

## Historical Context

### Original Work

The theorem was first published by:

!!! quote "Citation"
    
    T. S. Motzkin and E. G. Straus, "Maxima for graphs and a new proof of a theorem of Turán," *Canadian Journal of Mathematics*, vol. 17, pp. 533-540, 1965.

### Subsequent Developments

- **1970s**: Extensions to hypergraphs and weighted cases
- **1980s**: Computational studies and approximation algorithms  
- **1990s**: Semidefinite programming relaxations
- **2000s**: Modern optimization approaches (interior point, first-order methods)
- **2020s**: Quantum computing applications (our work with Dirac-3)

## Summary

The Motzkin-Straus theorem provides a beautiful mathematical framework that:

1. **Bridges Domains**: Connects discrete graph theory with continuous optimization
2. **Enables Algorithms**: Provides a computational pathway for NP-hard problems
3. **Offers Flexibility**: Supports various optimization backends and solver choices
4. **Maintains Rigor**: Preserves mathematical exactness through the equivalence

This theoretical foundation underlies our entire computational approach, enabling us to leverage powerful optimization tools—from classical solvers to quantum annealers—for solving maximum independent set problems.

---

**Next**: Explore the [specific algorithms](algorithms.md) that implement this theory, or dive into the [quadratic programming solvers](../api/oracles/qp-solvers.md) that make it computationally feasible.