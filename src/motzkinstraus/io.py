
import networkx as nx

def read_dimacs_graph(file_path: str) -> nx.Graph:
    """
    Read a graph from DIMACS format file.
    
    DIMACS format:
    - Lines starting with 'c' are comments
    - Line starting with 'p edge n m' defines problem with n nodes and m edges
    - Lines starting with 'e u v' define edges between nodes u and v
    
    Args:
        file_path: Path to DIMACS format file
        
    Returns:
        NetworkX graph representation
    """
    graph = nx.Graph()
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('c'):
                # Comment line, skip
                continue
            elif line.startswith('p edge'):
                # Problem definition: p edge <num_nodes> <num_edges>
                parts = line.split()
                if len(parts) >= 3:
                    num_nodes = int(parts[2])
                    # Add nodes 1 through num_nodes (DIMACS uses 1-based indexing)
                    graph.add_nodes_from(range(1, num_nodes + 1))
            elif line.startswith('e'):
                # Edge definition: e <node1> <node2>
                parts = line.split()
                if len(parts) >= 3:
                    u, v = int(parts[1]), int(parts[2])
                    graph.add_edge(u, v)
    
    return graph
