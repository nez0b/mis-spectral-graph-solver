
import networkx as nx
import motzkinstraus as ms
import matplotlib.pyplot as plt
from motzkinstraus.oracles.jax_pgd import ProjectedGradientDescentOracle

def solve_and_visualize_dimacs(file_path: str):
    """
    Reads a DIMACS graph, finds the max clique, and visualizes the result.
    """
    # Read the graph
    graph = ms.read_dimacs_graph(file_path)

    # Find the maximum clique
    # We find the max independent set of the complement graph
    complement_graph = nx.complement(graph)
    oracle = ProjectedGradientDescentOracle()
    clique, _ = ms.find_max_clique_with_oracle(complement_graph, oracle=oracle)

    # Print the result
    print(f"Max clique found: {clique}")
    print(f"Clique size: {len(clique)}")

    # Visualize the graph and the clique
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph, seed=42)
    node_colors = ['red' if node in clique else 'lightblue' for node in graph.nodes()]
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=10)
    plt.title(f"Max Clique (size {len(clique)}) in {file_path}")
    plt.savefig("dimacs_clique_solution.png")
    plt.show()

if __name__ == "__main__":
    solve_and_visualize_dimacs("DIMACS/test_10_node.dimacs")
