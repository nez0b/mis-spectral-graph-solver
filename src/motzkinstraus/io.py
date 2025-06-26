
import networkx as nx

def read_dimacs_graph(file_path: str) -> nx.Graph:
    """
    Reads a graph in DIMACS format from the specified file.

    Args:
        file_path: The path to the DIMACS file.

    Returns:
        A networkx graph.
    """
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('c'):  # comment
                continue
            elif line.startswith('p'):  # problem definition
                parts = line.split()
                num_nodes = int(parts[2])
                G.add_nodes_from(range(1, num_nodes + 1))
            elif line.startswith('e'):  # edge definition
                parts = line.split()
                u, v = int(parts[1]), int(parts[2])
                G.add_edge(u, v)
    return G
