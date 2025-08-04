#!/usr/bin/env python3
"""
Test whether a solution file meets a given density threshold.

Usage:
    python test_solution.py solution_file [--edges edges.csv] [--denominator 8]
    
Examples:
    python test_solution.py my_solution.txt
    python test_solution.py my_solution.txt --denominator 8
    python test_solution.py my_solution.txt --edges graph.csv --denominator 10
"""

import argparse
import sys
from collections import defaultdict

def load_solution(solution_file):
    """Load node IDs from solution file."""
    nodes = []
    try:
        with open(solution_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    nodes.append(int(line))
    except FileNotFoundError:
        print(f"Error: Solution file '{solution_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid node ID in solution file: {e}", file=sys.stderr)
        sys.exit(1)
    
    return nodes

def load_edges(edges_file):
    """Load edges from CSV file, return as adjacency set."""
    edges = defaultdict(set)
    edge_count = 0
    
    try:
        with open(edges_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    parts = line.split(',')
                    if len(parts) != 2:
                        print(f"Warning: Line {line_num} in edges file has {len(parts)} parts, expected 2. Skipping.", file=sys.stderr)
                        continue
                    
                    node1, node2 = int(parts[0].strip()), int(parts[1].strip())
                    
                    # Add bidirectional edges
                    edges[node1].add(node2)
                    edges[node2].add(node1)
                    edge_count += 1
                    
                except ValueError as e:
                    print(f"Warning: Invalid edge on line {line_num}: {line}. Error: {e}", file=sys.stderr)
                    continue
    
    except FileNotFoundError:
        print(f"Error: Edges file '{edges_file}' not found.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {edge_count} edges from {edges_file}", file=sys.stderr)
    return edges

def calculate_density(solution_nodes, edges):
    """Calculate the density of the solution subgraph."""
    n = len(solution_nodes)
    if n < 2:
        return 0.0
    
    solution_set = set(solution_nodes)
    edge_count = 0
    
    # Count edges within the solution subgraph
    for node in solution_nodes:
        if node in edges:
            # Count edges to other nodes in the solution
            neighbors_in_solution = edges[node].intersection(solution_set)
            edge_count += len(neighbors_in_solution)
    
    # Each edge is counted twice (once from each endpoint), so divide by 2
    edge_count //= 2
    
    # Density = actual_edges / possible_edges
    # For a clique of size n, there are n*(n-1)/2 possible edges
    max_possible_edges = n * (n - 1) // 2
    density = edge_count / max_possible_edges if max_possible_edges > 0 else 0.0
    
    return density

def main():
    parser = argparse.ArgumentParser(
        description="Test whether a solution file meets a given density threshold.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s solution.txt                           # Just print density
  %(prog)s solution.txt --denominator 8          # Check if density >= 1/8
  %(prog)s solution.txt --edges graph.csv -d 10  # Use custom edges file
        """
    )
    
    parser.add_argument('solution_file', 
                       help='File containing solution node IDs (one per line)')
    parser.add_argument('--edges', '-e', 
                       default='edges.csv',
                       help='Edges file (default: edges.csv)')
    parser.add_argument('--denominator', '-d', 
                       type=int,
                       help='Denominator for threshold (1/denominator). If not provided, only density is printed.')
    
    args = parser.parse_args()
    
    # Load solution and edges
    solution_nodes = load_solution(args.solution_file)
    edges = load_edges(args.edges)
    
    print(f"Solution contains {len(solution_nodes)} nodes", file=sys.stderr)
    
    # Check for duplicate nodes in solution
    unique_nodes = set(solution_nodes)
    if len(unique_nodes) != len(solution_nodes):
        print(f"Warning: Solution contains {len(solution_nodes) - len(unique_nodes)} duplicate nodes", file=sys.stderr)
        solution_nodes = list(unique_nodes)
    
    # Check if all solution nodes exist in the graph
    nodes_in_graph = set()
    for node_edges in edges.values():
        nodes_in_graph.update(node_edges)
    nodes_in_graph.update(edges.keys())
    
    missing_nodes = unique_nodes - nodes_in_graph
    if missing_nodes:
        print(f"Warning: {len(missing_nodes)} solution nodes not found in graph: {sorted(list(missing_nodes))[:10]}{'...' if len(missing_nodes) > 10 else ''}", file=sys.stderr)
    
    # Calculate density
    density = calculate_density(solution_nodes, edges)
    
    # Output results
    print(f"Density: {density:.8f}")
    
    if args.denominator is not None:
        threshold = 1.0 / args.denominator
        meets_threshold = density >= threshold
        print(f"Threshold (1/{args.denominator}): {threshold:.8f}")
        print(f"Meets threshold: {meets_threshold}")
        
        # Exit with appropriate code
        sys.exit(0 if meets_threshold else 1)

if __name__ == "__main__":
    main()