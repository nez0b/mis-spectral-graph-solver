#!/usr/bin/env python3
"""
C125.9 Clique Validation and Visualization Script

This script validates that the Dirac solver's MIS solution forms a valid clique 
in the original C125.9 graph and creates comprehensive visualizations.

Mathematical Foundation:
- MIS in complement(C125.9) = Max clique in C125.9
- Dirac found MIS with 34 nodes in complement
- These 34 nodes should form a clique in original C125.9
- Expected clique edges: 34 × 33 / 2 = 561 edges

Usage:
    python validate_c125_clique.py
"""

import os
import sys
import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Set, Tuple, Dict

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Import DIMACS reader
from demo_dimacs_clique_solvers import read_dimacs_graph

# Dirac solver's MIS solution from complement graph
# These nodes should form a clique in the original C125.9 graph
CLIQUE_NODES = [1, 2, 5, 7, 9, 11, 17, 18, 19, 24, 25, 29, 31, 34, 40, 44, 45, 47, 48, 49, 54, 70, 71, 77, 79, 80, 98, 101, 110, 115, 117, 121, 122, 125]

# Expected theoretical values
EXPECTED_CLIQUE_SIZE = 34
EXPECTED_CLIQUE_EDGES = EXPECTED_CLIQUE_SIZE * (EXPECTED_CLIQUE_SIZE - 1) // 2  # 561


def validate_clique(graph: nx.Graph, clique_nodes: list) -> Tuple[bool, Dict]:
    """
    Validate that the given nodes form a valid clique in the graph.
    
    Args:
        graph: NetworkX graph to check
        clique_nodes: List of node IDs that should form a clique
        
    Returns:
        Tuple of (is_valid_clique, validation_statistics)
    """
    print(f"\n{'='*60}")
    print("CLIQUE VALIDATION")
    print('='*60)
    
    clique_set = set(clique_nodes)
    n_clique = len(clique_nodes)
    
    # Basic checks
    print(f"Clique candidate: {n_clique} nodes")
    print(f"Expected clique size: {EXPECTED_CLIQUE_SIZE}")
    print(f"Expected clique edges: {EXPECTED_CLIQUE_EDGES}")
    
    # Check all nodes exist in graph
    graph_nodes = set(graph.nodes())
    missing_nodes = clique_set - graph_nodes
    if missing_nodes:
        print(f"❌ ERROR: Clique contains nodes not in graph: {missing_nodes}")
        return False, {"error": "Missing nodes", "missing_nodes": missing_nodes}
    
    print(f"✅ All clique nodes exist in graph")
    
    # Validate all pairwise connections
    print(f"🔍 Checking all pairwise connections...")
    
    missing_edges = []
    present_edges = []
    total_pairs = 0
    
    for i, node_u in enumerate(clique_nodes):
        for j, node_v in enumerate(clique_nodes):
            if i < j:  # Avoid double counting
                total_pairs += 1
                if graph.has_edge(node_u, node_v):
                    present_edges.append((node_u, node_v))
                else:
                    missing_edges.append((node_u, node_v))
    
    n_present = len(present_edges)
    n_missing = len(missing_edges)
    
    print(f"   Total pairs checked: {total_pairs}")
    print(f"   Present edges: {n_present}")
    print(f"   Missing edges: {n_missing}")
    
    is_valid = (n_missing == 0)
    
    if is_valid:
        print(f"✅ VALIDATION PASSED: Valid clique of size {n_clique}")
        print(f"✅ All {n_present} expected edges are present")
    else:
        print(f"❌ VALIDATION FAILED: Missing {n_missing} edges")
        print(f"   First few missing edges: {missing_edges[:10]}")
    
    # Additional statistics
    clique_subgraph = graph.subgraph(clique_nodes)
    actual_density = nx.density(clique_subgraph)
    
    validation_stats = {
        "is_valid": is_valid,
        "clique_size": n_clique,
        "expected_edges": EXPECTED_CLIQUE_EDGES,
        "present_edges": n_present,
        "missing_edges": n_missing,
        "missing_edge_list": missing_edges,
        "clique_density": actual_density,
        "theoretical_density": 1.0,
        "validation_time": time.time()
    }
    
    return is_valid, validation_stats


def analyze_clique_properties(graph: nx.Graph, clique_nodes: list) -> Dict:
    """
    Analyze detailed properties of the clique.
    
    Args:
        graph: NetworkX graph
        clique_nodes: List of clique node IDs
        
    Returns:
        Dictionary of clique analysis results
    """
    print(f"\n{'='*60}")
    print("CLIQUE ANALYSIS")
    print('='*60)
    
    clique_subgraph = graph.subgraph(clique_nodes)
    
    # Basic properties
    n_nodes = clique_subgraph.number_of_nodes()
    n_edges = clique_subgraph.number_of_edges()
    density = nx.density(clique_subgraph)
    
    print(f"📊 Clique Properties:")
    print(f"   Nodes: {n_nodes}")
    print(f"   Edges: {n_edges}")
    print(f"   Density: {density:.6f} (expected: 1.000000)")
    print(f"   Complete: {'Yes' if density == 1.0 else 'No'}")
    
    # Degree analysis
    degrees = dict(clique_subgraph.degree())
    min_degree = min(degrees.values()) if degrees else 0
    max_degree = max(degrees.values()) if degrees else 0
    avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
    expected_degree = n_nodes - 1  # In a clique, each node connects to all others
    
    print(f"\n📊 Degree Analysis:")
    print(f"   Expected degree: {expected_degree}")
    print(f"   Min degree: {min_degree}")
    print(f"   Max degree: {max_degree}")
    print(f"   Average degree: {avg_degree:.2f}")
    print(f"   Degree consistency: {'✅' if min_degree == max_degree == expected_degree else '❌'}")
    
    # Clique nodes in original graph context
    print(f"\n📊 Context in Original Graph:")
    original_n = graph.number_of_nodes()
    original_m = graph.number_of_edges()
    clique_fraction = n_nodes / original_n
    edge_fraction = n_edges / original_m if original_m > 0 else 0
    
    print(f"   Original graph: {original_n} nodes, {original_m} edges")
    print(f"   Clique nodes: {n_nodes}/{original_n} ({clique_fraction:.3%})")
    print(f"   Clique edges: {n_edges}/{original_m} ({edge_fraction:.3%})")
    
    # Theoretical validation
    max_possible_edges = original_n * (original_n - 1) // 2
    complement_edges = max_possible_edges - original_m
    
    print(f"\n🔍 Theoretical Validation:")
    print(f"   Max possible edges in {original_n}-node graph: {max_possible_edges}")
    print(f"   Original graph edges: {original_m}")
    print(f"   Complement graph edges: {complement_edges}")
    print(f"   Edge sum validation: {original_m + complement_edges == max_possible_edges}")
    
    analysis_results = {
        "clique_nodes": n_nodes,
        "clique_edges": n_edges,
        "clique_density": density,
        "degrees": degrees,
        "min_degree": min_degree,
        "max_degree": max_degree,
        "avg_degree": avg_degree,
        "expected_degree": expected_degree,
        "degree_consistent": min_degree == max_degree == expected_degree,
        "original_nodes": original_n,
        "original_edges": original_m,
        "clique_node_fraction": clique_fraction,
        "clique_edge_fraction": edge_fraction,
        "complement_edges": complement_edges,
        "theoretical_valid": original_m + complement_edges == max_possible_edges
    }
    
    return analysis_results


def create_clique_visualization(graph: nx.Graph, clique_nodes: list, output_dir: Path):
    """
    Create a two-panel visualization showing the clique in context and detail.
    
    Args:
        graph: NetworkX graph
        clique_nodes: List of clique node IDs
        output_dir: Directory to save plots
    """
    print(f"\n{'='*60}")
    print("CREATING VISUALIZATIONS")
    print('='*60)
    
    clique_set = set(clique_nodes)
    clique_subgraph = graph.subgraph(clique_nodes)
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Panel 1: Context view - Full graph with highlighted clique
    print("📊 Creating context view (full graph with clique highlight)...")
    
    # Use circular layout for full graph (handles 125 nodes well)
    pos_full = nx.circular_layout(graph, scale=2)
    
    # Node colors: red for clique, light gray for others
    node_colors_full = ['red' if node in clique_set else 'lightgray' for node in graph.nodes()]
    
    # Node sizes: larger for clique nodes
    node_sizes_full = [200 if node in clique_set else 30 for node in graph.nodes()]
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos_full, 
                          node_color=node_colors_full,
                          node_size=node_sizes_full,
                          alpha=0.8,
                          ax=ax1)
    
    # Draw edges with minimal style for background, highlighted for clique
    print("   Drawing edges (this may take a moment for dense graph)...")
    
    # Draw all edges very faintly
    nx.draw_networkx_edges(graph, pos_full,
                          alpha=0.05,
                          width=0.1,
                          edge_color='gray',
                          ax=ax1)
    
    # Highlight clique edges
    clique_edges = []
    for u in clique_nodes:
        for v in clique_nodes:
            if u < v and graph.has_edge(u, v):
                clique_edges.append((u, v))
    
    if clique_edges:
        nx.draw_networkx_edges(graph, pos_full,
                              edgelist=clique_edges,
                              alpha=0.6,
                              width=1.5,
                              edge_color='red',
                              ax=ax1)
    
    ax1.set_title(f'C125.9 Graph with Highlighted Clique\n'
                  f'Total: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges\n'
                  f'Clique: {len(clique_nodes)} nodes (red), {len(clique_edges)} edges',
                  fontsize=12)
    ax1.axis('off')
    
    # Panel 2: Detail view - Clique subgraph only
    print("📊 Creating detail view (clique subgraph only)...")
    
    # Use spring layout for clique (shows structure well)
    pos_clique = nx.spring_layout(clique_subgraph, k=3, iterations=100, seed=42)
    
    # Draw clique nodes
    nx.draw_networkx_nodes(clique_subgraph, pos_clique,
                          node_color='red',
                          node_size=400,
                          alpha=0.9,
                          ax=ax2)
    
    # Draw clique edges
    nx.draw_networkx_edges(clique_subgraph, pos_clique,
                          alpha=0.7,
                          width=2.0,
                          edge_color='darkred',
                          ax=ax2)
    
    # Add node labels for clique
    nx.draw_networkx_labels(clique_subgraph, pos_clique,
                           font_size=8,
                           font_color='white',
                           font_weight='bold',
                           ax=ax2)
    
    clique_density = nx.density(clique_subgraph)
    ax2.set_title(f'Clique Detail View\n'
                  f'{len(clique_nodes)} nodes, {clique_subgraph.number_of_edges()} edges\n'
                  f'Density: {clique_density:.6f} (Complete: {"Yes" if clique_density == 1.0 else "No"})',
                  fontsize=12)
    ax2.axis('off')
    
    # Overall title
    plt.suptitle('C125.9 Maximum Clique Validation\n'
                 'Found via MIS on Complement Graph (Dirac Solver)',
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / 'c125_clique_validation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_path}")
    
    plt.show()


def print_validation_results(is_valid: bool, validation_stats: Dict, analysis_results: Dict):
    """
    Print comprehensive validation results.
    
    Args:
        is_valid: Whether the clique validation passed
        validation_stats: Validation statistics
        analysis_results: Clique analysis results
    """
    print(f"\n{'='*80}")
    print("FINAL VALIDATION SUMMARY")
    print('='*80)
    
    print(f"🎯 CLIQUE VALIDATION: {'✅ PASSED' if is_valid else '❌ FAILED'}")
    
    if is_valid:
        print(f"✅ Successfully validated clique of size {validation_stats['clique_size']}")
        print(f"✅ All {validation_stats['present_edges']} expected edges confirmed")
        print(f"✅ Clique density: {validation_stats['clique_density']:.6f}")
        print(f"✅ Degree consistency: {'Yes' if analysis_results['degree_consistent'] else 'No'}")
    else:
        print(f"❌ Validation failed: {validation_stats['missing_edges']} missing edges")
        print(f"❌ Only {validation_stats['present_edges']}/{validation_stats['expected_edges']} edges found")
    
    print(f"\n📊 THEORETICAL VALIDATION:")
    print(f"   MIS→Clique transformation: {'✅ Confirmed' if is_valid else '❌ Failed'}")
    print(f"   Graph complement property: {'✅ Valid' if analysis_results['theoretical_valid'] else '❌ Invalid'}")
    print(f"   Dirac solver accuracy: {'✅ Correct' if is_valid else '❌ Incorrect'}")
    
    print(f"\n📊 CLIQUE PROPERTIES:")
    print(f"   Size: {analysis_results['clique_nodes']} nodes")
    print(f"   Edges: {analysis_results['clique_edges']} edges")
    print(f"   Fraction of graph: {analysis_results['clique_node_fraction']:.1%} nodes, {analysis_results['clique_edge_fraction']:.1%} edges")
    
    if is_valid:
        print(f"\n🎉 SUCCESS: Dirac solver correctly identified maximum clique of size {validation_stats['clique_size']} in C125.9!")
    else:
        print(f"\n⚠️  WARNING: Dirac solver result does not form a valid clique in original graph")


def generate_analysis_report(validation_stats: Dict, analysis_results: Dict, output_dir: Path):
    """
    Generate a detailed text report of the validation and analysis.
    
    Args:
        validation_stats: Validation statistics
        analysis_results: Clique analysis results
        output_dir: Directory to save report
    """
    report_path = output_dir / 'c125_clique_analysis.txt'
    
    with open(report_path, 'w') as f:
        f.write("C125.9 Maximum Clique Validation Report\n")
        f.write("="*50 + "\n\n")
        
        f.write("MATHEMATICAL FOUNDATION:\n")
        f.write("- Maximum clique in G = MIS in complement(G)\n")
        f.write("- Dirac solver found MIS in C125.9 complement\n")
        f.write("- These nodes should form clique in original C125.9\n\n")
        
        f.write("VALIDATION RESULTS:\n")
        f.write(f"- Clique validation: {'PASSED' if validation_stats['is_valid'] else 'FAILED'}\n")
        f.write(f"- Clique size: {validation_stats['clique_size']} nodes\n")
        f.write(f"- Expected edges: {validation_stats['expected_edges']}\n")
        f.write(f"- Present edges: {validation_stats['present_edges']}\n")
        f.write(f"- Missing edges: {validation_stats['missing_edges']}\n")
        f.write(f"- Clique density: {validation_stats['clique_density']:.6f}\n\n")
        
        f.write("CLIQUE ANALYSIS:\n")
        f.write(f"- Nodes: {analysis_results['clique_nodes']}\n")
        f.write(f"- Edges: {analysis_results['clique_edges']}\n")
        f.write(f"- Degree consistency: {analysis_results['degree_consistent']}\n")
        f.write(f"- Node fraction: {analysis_results['clique_node_fraction']:.3%}\n")
        f.write(f"- Edge fraction: {analysis_results['clique_edge_fraction']:.3%}\n\n")
        
        f.write("THEORETICAL VALIDATION:\n")
        f.write(f"- Original graph: {analysis_results['original_nodes']} nodes, {analysis_results['original_edges']} edges\n")
        f.write(f"- Complement graph: {analysis_results['original_nodes']} nodes, {analysis_results['complement_edges']} edges\n")
        f.write(f"- Edge sum correct: {analysis_results['theoretical_valid']}\n\n")
        
        f.write("CLIQUE NODES:\n")
        f.write(f"{CLIQUE_NODES}\n\n")
        
        if validation_stats['missing_edges']:
            f.write("MISSING EDGES (if any):\n")
            for u, v in validation_stats['missing_edge_list'][:20]:  # First 20
                f.write(f"  ({u}, {v})\n")
            if len(validation_stats['missing_edge_list']) > 20:
                f.write(f"  ... and {len(validation_stats['missing_edge_list']) - 20} more\n")
    
    print(f"📄 Analysis report saved to: {report_path}")


def main():
    """
    Main function to validate and visualize the C125.9 clique.
    """
    print("🎯 C125.9 Clique Validation and Visualization")
    print("="*60)
    print("Validating Dirac solver's MIS solution as clique in original graph")
    print(f"Candidate clique: {len(CLIQUE_NODES)} nodes")
    print(f"Expected edges: {EXPECTED_CLIQUE_EDGES}")
    
    # Create output directory
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    
    try:
        # Load C125.9 graph
        print(f"\n📁 Loading C125.9 graph...")
        graph = read_dimacs_graph("DIMACS/C125.9.clq")
        print(f"✅ Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Validate clique
        is_valid, validation_stats = validate_clique(graph, CLIQUE_NODES)
        
        # Analyze clique properties
        analysis_results = analyze_clique_properties(graph, CLIQUE_NODES)
        
        # Create visualizations
        create_clique_visualization(graph, CLIQUE_NODES, output_dir)
        
        # Generate report
        generate_analysis_report(validation_stats, analysis_results, output_dir)
        
        # Print final results
        print_validation_results(is_valid, validation_stats, analysis_results)
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Analysis completed in {total_time:.2f} seconds")
        print(f"📊 All outputs saved to: {output_dir.absolute()}")
        
        return 0 if is_valid else 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())