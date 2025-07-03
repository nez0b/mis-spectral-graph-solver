#!/usr/bin/env python3
"""
Graph Analysis and Visualization Script

This script reads a graph in DIMACS format and creates side-by-side visualizations
of the original graph and its complement, with detailed statistics.

Usage:
    python analyze_graph.py DIMACS/graph_file.dimacs
"""

import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import DIMACS reader
from motzkinstraus.io import read_dimacs_graph


def analyze_graph_properties(graph, name):
    """Analyze and return comprehensive graph properties."""
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    
    properties = {
        'name': name,
        'nodes': n,
        'edges': m,
        'density': m / (n * (n - 1) / 2) if n > 1 else 0,
        'avg_degree': 2 * m / n if n > 0 else 0,
        'max_degree': max(dict(graph.degree()).values()) if n > 0 else 0,
        'min_degree': min(dict(graph.degree()).values()) if n > 0 else 0,
        'is_connected': nx.is_connected(graph) if n > 0 else False,
        'num_components': nx.number_connected_components(graph),
        'clustering_coeff': nx.average_clustering(graph) if n > 0 and n <= 200 else "N/A (too large)"
    }
    
    # Add diameter for connected graphs (expensive for large graphs)
    if properties['is_connected'] and n <= 50:
        try:
            properties['diameter'] = nx.diameter(graph)
        except:
            properties['diameter'] = "N/A"
    else:
        properties['diameter'] = "N/A (disconnected or too large)"
    
    return properties


def create_graph_layout(graph, max_nodes_for_spring=50):
    """Create an appropriate layout for graph visualization."""
    n = graph.number_of_nodes()
    
    if n <= max_nodes_for_spring:
        # Use spring layout for smaller graphs
        return nx.spring_layout(graph, k=1, iterations=50, seed=42)
    else:
        # Use circular layout for larger graphs
        return nx.circular_layout(graph)


def plot_graph_comparison(graph, graph_complement, graph_name, output_dir):
    """Create side-by-side comparison plots of graph and its complement."""
    
    # Analyze properties
    props = analyze_graph_properties(graph, graph_name)
    props_comp = analyze_graph_properties(graph_complement, f"{graph_name} (Complement)")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Determine visualization approach based on graph size
    n = graph.number_of_nodes()
    
    if n <= 30:
        # Full visualization for small graphs
        pos = create_graph_layout(graph)
        
        # Plot original graph
        nx.draw_networkx_nodes(graph, pos, node_color='lightblue', 
                              node_size=300, ax=ax1)
        nx.draw_networkx_edges(graph, pos, alpha=0.5, ax=ax1)
        nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax1)
        
        # Plot complement graph
        nx.draw_networkx_nodes(graph_complement, pos, node_color='lightcoral', 
                              node_size=300, ax=ax2)
        nx.draw_networkx_edges(graph_complement, pos, alpha=0.5, ax=ax2)
        nx.draw_networkx_labels(graph_complement, pos, font_size=8, ax=ax2)
        
    elif n <= 100:
        # Node-only visualization for medium graphs
        pos = create_graph_layout(graph)
        
        # Plot original graph
        nx.draw_networkx_nodes(graph, pos, node_color='lightblue', 
                              node_size=50, ax=ax1)
        nx.draw_networkx_edges(graph, pos, alpha=0.3, width=0.5, ax=ax1)
        
        # Plot complement graph  
        nx.draw_networkx_nodes(graph_complement, pos, node_color='lightcoral', 
                              node_size=50, ax=ax2)
        nx.draw_networkx_edges(graph_complement, pos, alpha=0.3, width=0.5, ax=ax2)
        
    else:
        # Graph visualization for large graphs with small nodes
        pos = create_graph_layout(graph, max_nodes_for_spring=0)  # Force circular layout
        
        # Calculate appropriate node size based on graph size
        node_size = max(10, 500 / n)  # Smaller nodes for larger graphs
        edge_alpha = max(0.1, 50 / n)  # More transparent edges for larger graphs
        edge_width = max(0.1, 20 / n)  # Thinner edges for larger graphs
        
        # Plot original graph
        nx.draw_networkx_nodes(graph, pos, node_color='lightblue', 
                              node_size=node_size, ax=ax1)
        nx.draw_networkx_edges(graph, pos, alpha=edge_alpha, width=edge_width, ax=ax1)
        
        # Plot complement graph  
        nx.draw_networkx_nodes(graph_complement, pos, node_color='lightcoral', 
                              node_size=node_size, ax=ax2)
        nx.draw_networkx_edges(graph_complement, pos, alpha=edge_alpha, width=edge_width, ax=ax2)
    
    # Set titles for small/medium graphs
    title1 = f'Original Graph\nNodes: {props["nodes"]}, Edges: {props["edges"]}\nDensity: {props["density"]:.3f}'
    title2 = f'Complement Graph\nNodes: {props_comp["nodes"]}, Edges: {props_comp["edges"]}\nDensity: {props_comp["density"]:.3f}'
    
    ax1.set_title(title1)
    ax2.set_title(title2)
    ax1.axis('off')
    ax2.axis('off')
    
    plt.suptitle(f'{graph_name} - Graph vs Complement Comparison', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    filename = graph_name.lower().replace(' ', '_').replace('.', '_')
    plt.savefig(output_dir / f'{filename}_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return props, props_comp


def print_detailed_analysis(props, props_comp):
    """Print detailed analysis of both graphs."""
    print(f"\n{'='*80}")
    print("DETAILED GRAPH ANALYSIS")
    print('='*80)
    
    print(f"\n📊 ORIGINAL GRAPH: {props['name']}")
    print(f"   Nodes: {props['nodes']}")
    print(f"   Edges: {props['edges']}")
    print(f"   Density: {props['density']:.4f}")
    print(f"   Average Degree: {props['avg_degree']:.2f}")
    print(f"   Degree Range: {props['min_degree']} - {props['max_degree']}")
    print(f"   Connected: {'Yes' if props['is_connected'] else 'No'}")
    print(f"   Components: {props['num_components']}")
    print(f"   Clustering Coefficient: {props['clustering_coeff']:.4f}")
    print(f"   Diameter: {props['diameter']}")
    
    print(f"\n📊 COMPLEMENT GRAPH: {props_comp['name']}")
    print(f"   Nodes: {props_comp['nodes']}")
    print(f"   Edges: {props_comp['edges']}")
    print(f"   Density: {props_comp['density']:.4f}")
    print(f"   Average Degree: {props_comp['avg_degree']:.2f}")
    print(f"   Degree Range: {props_comp['min_degree']} - {props_comp['max_degree']}")
    print(f"   Connected: {'Yes' if props_comp['is_connected'] else 'No'}")
    print(f"   Components: {props_comp['num_components']}")
    print(f"   Clustering Coefficient: {props_comp['clustering_coeff']:.4f}")
    print(f"   Diameter: {props_comp['diameter']}")
    
    # Theoretical validation
    n = props['nodes']
    max_edges = n * (n - 1) // 2
    total_edges = props['edges'] + props_comp['edges']
    
    print(f"\n🔍 THEORETICAL VALIDATION:")
    print(f"   Maximum possible edges: {max_edges}")
    print(f"   Original + Complement edges: {props['edges']} + {props_comp['edges']} = {total_edges}")
    print(f"   Validation: {'✅ PASS' if total_edges == max_edges else '❌ FAIL'}")
    print(f"   Density sum: {props['density']:.4f} + {props_comp['density']:.4f} = {props['density'] + props_comp['density']:.4f}")
    print(f"   Expected density sum: 1.000 {'✅' if abs(props['density'] + props_comp['density'] - 1.0) < 1e-10 else '❌'}")


def main():
    """Main function to analyze and visualize a graph and its complement."""
    parser = argparse.ArgumentParser(
        description="Analyze and visualize a graph and its complement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_graph.py DIMACS/C125.9.clq
  python analyze_graph.py DIMACS/complete_k5.dimacs
  python analyze_graph.py DIMACS/erdos_renyi_15_p05_seed123.dimacs
        """
    )
    parser.add_argument("dimacs_file", help="Path to DIMACS format file")
    parser.add_argument("--output-dir", default="figures", help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("📈 Graph Analysis and Visualization")
    print("=" * 60)
    
    # Check file exists
    if not os.path.exists(args.dimacs_file):
        print(f"❌ DIMACS file not found: {args.dimacs_file}")
        return 1
    
    try:
        # Read the graph
        print(f"📁 Reading DIMACS file: {args.dimacs_file}")
        graph = read_dimacs_graph(args.dimacs_file)
        graph_name = Path(args.dimacs_file).stem
        
        print(f"✅ Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Create complement graph
        print("🔄 Computing complement graph...")
        graph_complement = nx.complement(graph)
        
        print(f"✅ Complement computed: {graph_complement.number_of_nodes()} nodes, {graph_complement.number_of_edges()} edges")
        
        # Create visualization and analysis
        print("📊 Creating visualizations...")
        props, props_comp = plot_graph_comparison(graph, graph_complement, graph_name, output_dir)
        
        # Print detailed analysis
        print_detailed_analysis(props, props_comp)
        
        print(f"\n🎉 Analysis completed successfully!")
        print(f"📊 Visualization saved to: {output_dir.absolute()}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())