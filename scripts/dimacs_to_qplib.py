#!/usr/bin/env python3
"""
Convert DIMACS format graphs to QPLIB JSON format for Motzkin-Straus optimization.

This script reads DIMACS format graph files and converts them to QPLIB JSON format
suitable for quadratic programming. For the Motzkin-Straus theorem, edges are
represented as polynomial terms with coefficients of 1.0.

Usage:
    python scripts/dimacs_to_qplib.py input.dimacs output.json [--validate]

Example:
    python scripts/dimacs_to_qplib.py DIMACS/test_10_node.dimacs test_output.json
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import networkx as nx
    from motzkinstraus.io import read_dimacs_graph
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure NetworkX and motzkinstraus are installed.")
    sys.exit(1)


def dimacs_to_qplib(dimacs_file: str) -> Dict[str, Any]:
    """
    Convert a DIMACS format graph file to QPLIB JSON format.
    
    For the Motzkin-Straus quadratic program, we want to maximize:
    0.5 * x^T * A * x subject to sum(x_i) = 1, x_i >= 0
    
    This expands to: 0.5 * sum_{(i,j) in E} x_i * x_j
    
    In QPLIB format:
    - poly_indices: List of [i, j] pairs representing edges
    - poly_coefficients: List of 1.0 values (one per edge)  
    - sum_constraint: Always 1 for Motzkin-Straus
    
    Args:
        dimacs_file: Path to DIMACS format graph file
        
    Returns:
        Dictionary in QPLIB format with keys: poly_indices, poly_coefficients, sum_constraint
        
    Raises:
        FileNotFoundError: If the DIMACS file doesn't exist
        ValueError: If the graph cannot be processed
    """
    if not os.path.exists(dimacs_file):
        raise FileNotFoundError(f"DIMACS file not found: {dimacs_file}")
    
    try:
        # Read DIMACS file using existing utility
        graph = read_dimacs_graph(dimacs_file)
        
        if graph.number_of_nodes() == 0:
            # Handle empty graph
            return {
                "poly_indices": [],
                "poly_coefficients": [],
                "sum_constraint": 1
            }
        
        # Extract edges and convert to polynomial format
        poly_indices = []
        poly_coefficients = []
        
        # For undirected graphs, include each edge once as [i, j] where i <= j
        # This avoids double-counting edges in the quadratic form
        for u, v in graph.edges():
            # Ensure consistent ordering for undirected edges
            if u <= v:
                poly_indices.append([u, v])
                poly_coefficients.append(1.0)
            else:
                poly_indices.append([v, u])
                poly_coefficients.append(1.0)
        
        # Include self-loops if present (diagonal terms)
        for node in graph.nodes():
            if graph.has_edge(node, node):
                poly_indices.append([node, node])
                poly_coefficients.append(1.0)
        
        return {
            "poly_indices": poly_indices,
            "poly_coefficients": poly_coefficients,
            "sum_constraint": 1
        }
        
    except Exception as e:
        raise ValueError(f"Error processing DIMACS file {dimacs_file}: {e}")


def validate_qplib_format(qplib_data: Dict[str, Any]) -> bool:
    """
    Validate that the QPLIB data has the correct format.
    
    Args:
        qplib_data: Dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = {"poly_indices", "poly_coefficients", "sum_constraint"}
    
    # Check required keys
    if not all(key in qplib_data for key in required_keys):
        return False
    
    # Check types
    if not isinstance(qplib_data["poly_indices"], list):
        return False
    if not isinstance(qplib_data["poly_coefficients"], list):
        return False
    if not isinstance(qplib_data["sum_constraint"], (int, float)):
        return False
    
    # Check lengths match
    if len(qplib_data["poly_indices"]) != len(qplib_data["poly_coefficients"]):
        return False
    
    # Check poly_indices format
    for indices in qplib_data["poly_indices"]:
        if not isinstance(indices, list) or len(indices) != 2:
            return False
        if not all(isinstance(idx, int) and idx >= 1 for idx in indices):
            return False
    
    # Check poly_coefficients format
    if not all(isinstance(coeff, (int, float)) for coeff in qplib_data["poly_coefficients"]):
        return False
    
    return True


def print_conversion_summary(dimacs_file: str, qplib_data: Dict[str, Any], graph: nx.Graph = None) -> None:
    """Print a summary of the conversion."""
    print(f"\n{'='*60}")
    print(f"DIMACS to QPLIB Conversion Summary")
    print(f"{'='*60}")
    print(f"Input file: {dimacs_file}")
    
    if graph is not None:
        print(f"Graph properties:")
        print(f"  Nodes: {graph.number_of_nodes()}")
        print(f"  Edges: {graph.number_of_edges()}")
        print(f"  Density: {nx.density(graph):.4f}")
    
    print(f"QPLIB output:")
    print(f"  Polynomial terms: {len(qplib_data['poly_indices'])}")
    print(f"  Sum constraint: {qplib_data['sum_constraint']}")
    
    if qplib_data["poly_indices"]:
        print(f"  Index range: [{min(min(indices) for indices in qplib_data['poly_indices'])}, "
              f"{max(max(indices) for indices in qplib_data['poly_indices'])}]")
        print(f"  Coefficient range: [{min(qplib_data['poly_coefficients']):.3f}, "
              f"{max(qplib_data['poly_coefficients']):.3f}]")
    
    print(f"{'='*60}")


def main():
    """Main function to handle command-line interface and execution."""
    parser = argparse.ArgumentParser(
        description="Convert DIMACS graph files to QPLIB JSON format for Motzkin-Straus optimization",
        epilog="Example: python scripts/dimacs_to_qplib.py DIMACS/test_10_node.dimacs output.json"
    )
    
    parser.add_argument(
        "input", 
        help="Input DIMACS file path"
    )
    parser.add_argument(
        "output", 
        help="Output QPLIB JSON file path"
    )
    parser.add_argument(
        "--validate", 
        action="store_true",
        help="Validate output format before saving"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed conversion information"
    )
    
    args = parser.parse_args()
    
    try:
        # Convert DIMACS to QPLIB format
        if args.verbose:
            print(f"Reading DIMACS file: {args.input}")
        
        qplib_data = dimacs_to_qplib(args.input)
        
        # Optional validation
        if args.validate:
            if args.verbose:
                print("Validating QPLIB format...")
            
            if not validate_qplib_format(qplib_data):
                print("ERROR: Generated QPLIB data failed validation", file=sys.stderr)
                sys.exit(1)
            
            if args.verbose:
                print("✓ QPLIB format validation passed")
        
        # Create output directory if needed
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save QPLIB JSON
        with open(args.output, 'w') as f:
            json.dump(qplib_data, f, indent=2)
        
        if args.verbose:
            # Read the graph again for summary (could be optimized)
            try:
                graph = read_dimacs_graph(args.input)
                print_conversion_summary(args.input, qplib_data, graph)
            except:
                print_conversion_summary(args.input, qplib_data)
        
        print(f"Successfully converted {args.input} to {args.output}")
        
        if not args.verbose:
            print(f"Converted {len(qplib_data['poly_indices'])} polynomial terms")
    
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error during conversion: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()