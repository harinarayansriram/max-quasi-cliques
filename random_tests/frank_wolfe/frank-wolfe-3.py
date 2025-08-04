import numpy as np
import scipy.sparse as sp
import networkx as nx
import time
import random
from collections import defaultdict
import sys
from tqdm import tqdm

def load_graph(filename):
    """Load graph from edges file."""
    print("Loading graph from file...")
    G = nx.Graph()
    
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():
                u, v = line.strip().split(',')
                u, v = int(u), int(v)
                G.add_edge(u, v)
    
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def is_mostly_clique(G, nodes):
    """Check if the subgraph induced by nodes is a mostly-clique."""
    if len(nodes) <= 1:
        return True
        
    subgraph = G.subgraph(nodes)
    n = len(nodes)
    m = subgraph.number_of_edges()
    max_edges = n * (n - 1) / 2
    
    return m > max_edges / 2

def compute_density(G, nodes):
    """Compute density of the subgraph induced by nodes."""
    if len(nodes) <= 1:
        return 1.0
        
    subgraph = G.subgraph(nodes)
    n = len(nodes)
    m = subgraph.number_of_edges()
    max_edges = n * (n - 1) / 2
    
    return m / max_edges

def greedy_expansion(G, seed_nodes, max_iterations=10000):
    """Greedily expand the solution starting from seed nodes."""
    current_nodes = set(seed_nodes)
    if not current_nodes:
        # Start with the highest degree node if no seed is provided
        current_nodes = {max(G.nodes(), key=lambda n: G.degree(n))}
    
    candidates = set(G.nodes()) - current_nodes
    improved = True
    iteration = 0
    
    print(f"Starting greedy expansion from {len(current_nodes)} nodes")
    best_size_so_far = len(current_nodes)
    last_improvement = 0
    
    while improved and iteration < max_iterations:
        iteration += 1
        improved = False
        
        # Calculate connections to the current set for all candidates
        connections = {}
        for node in candidates:
            connections[node] = sum(1 for neighbor in G.neighbors(node) if neighbor in current_nodes)
        
        # Sort candidates by their connections (most connected first)
        sorted_candidates = sorted(connections.items(), key=lambda x: x[1], reverse=True)
        
        # Try adding the best candidates
        for node, _ in sorted_candidates[:100]:  # Try top 100 candidates for efficiency
            test_nodes = current_nodes | {node}
            if is_mostly_clique(G, test_nodes):
                current_nodes.add(node)
                candidates.remove(node)
                improved = True
                
                if len(current_nodes) > best_size_so_far:
                    best_size_so_far = len(current_nodes)
                    last_improvement = iteration
                    print(f"Improved to size {best_size_so_far} (iteration {iteration})")
                
                break
        
        # If no improvement for too long, break
        if iteration - last_improvement > 1000:
            print("No improvement for 1000 iterations, stopping")
            break
    
    density = compute_density(G, current_nodes)
    print(f"Greedy expansion result: size = {len(current_nodes)}, density = {density:.4f}")
    return list(current_nodes)

def k_core_based_solution(G, min_density=0.5):
    """Use k-core decomposition to find dense subgraphs, then expand."""
    print("Computing k-core decomposition...")
    core_numbers = nx.core_number(G)
    max_core = max(core_numbers.values())
    
    print(f"Maximum core number: {max_core}")
    
    # Group nodes by core number
    cores = defaultdict(list)
    for node, core in core_numbers.items():
        cores[core].append(node)
    
    best_size = 0
    best_solution = []
    
    # Check each k-core, starting from the highest
    for k in range(max_core, 0, -1):
        if k < max_core // 2 and best_size > 0:
            # Stop early if we've found something and are checking much smaller cores
            break
            
        k_core_nodes = []
        for c in range(k, max_core + 1):
            k_core_nodes.extend(cores[c])
        
        if not k_core_nodes:
            continue
            
        print(f"Checking {k}-core with {len(k_core_nodes)} nodes")
        
        # Check if this k-core is already a mostly-clique
        if is_mostly_clique(G, k_core_nodes):
            print(f"Found {k}-core with {len(k_core_nodes)} nodes that is a mostly-clique")
            
            # Try to expand it
            expanded = greedy_expansion(G, k_core_nodes)
            
            if len(expanded) > best_size:
                best_size = len(expanded)
                best_solution = expanded
                print(f"New best solution: size = {best_size}")
    
    return best_solution

def random_edge_induction(G, num_seeds=10, min_density=0.5):
    """Start from random edges and induce subgraphs."""
    best_size = 0
    best_solution = []
    
    for i in range(num_seeds):
        print(f"Random seed {i+1}/{num_seeds}")
        
        # Pick a random edge
        edges = list(G.edges())
        if not edges:
            continue
            
        edge = random.choice(edges)
        seed_nodes = list(edge)
        
        # Get common neighbors
        common_neighbors = set(G.neighbors(edge[0])) & set(G.neighbors(edge[1]))
        seed_nodes.extend(list(common_neighbors)[:10])  # Limit to 10 common neighbors
        
        # Expand
        expanded = greedy_expansion(G, seed_nodes)
        
        if len(expanded) > best_size:
            best_size = len(expanded)
            best_solution = expanded
            print(f"New best solution: size = {best_size}")
    
    return best_solution

def degree_based_solution(G, min_density=0.5):
    """Start from high-degree nodes and expand."""
    best_size = 0
    best_solution = []
    
    # Try top 5% of nodes by degree
    nodes_by_degree = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)
    top_nodes = nodes_by_degree[:max(10, len(nodes_by_degree) // 20)]
    
    for i, node in enumerate(top_nodes):
        print(f"Trying high-degree node {i+1}/{len(top_nodes)}")
        
        # Take the node and its highest-degree neighbors
        neighbors = sorted(G.neighbors(node), key=lambda n: G.degree(n), reverse=True)
        seed_nodes = [node] + neighbors[:min(50, len(neighbors))]
        
        # Check if seed is a mostly-clique, if not, take a subset
        while len(seed_nodes) > 1 and not is_mostly_clique(G, seed_nodes):
            seed_nodes = seed_nodes[:-1]
        
        # Expand
        expanded = greedy_expansion(G, seed_nodes)
        
        if len(expanded) > best_size:
            best_size = len(expanded)
            best_solution = expanded
            print(f"New best solution: size = {best_size}")
    
    return best_solution

def solve_mostly_clique(graph_file, output_file):
    """Main solver function combining multiple approaches."""
    start_time = time.time()
    
    # Load graph
    G = load_graph(graph_file)
    
    # Define different solution strategies
    solutions = []
    
    # Try k-core decomposition
    print("\n=== K-Core Based Solution ===")
    k_core_solution = k_core_based_solution(G)
    solutions.append((len(k_core_solution), k_core_solution, "k-core"))
    
    # Try degree-based solution
    print("\n=== Degree-Based Solution ===")
    degree_solution = degree_based_solution(G)
    solutions.append((len(degree_solution), degree_solution, "degree"))
    
    # Try random edge induction
    print("\n=== Random Edge Induction ===")
    random_solution = random_edge_induction(G, num_seeds=5)
    solutions.append((len(random_solution), random_solution, "random"))
    
    # Find the best solution
    solutions.sort(reverse=True)
    best_size, best_solution, method = solutions[0]
    
    print("\n=== Results ===")
    for size, sol, method in solutions:
        print(f"{method}: size = {size}, density = {compute_density(G, sol):.4f}")
    
    print(f"\nBest solution from {method} approach: size = {best_size}")
    
    # Write solution to file
    with open(output_file, 'w') as f:
        f.write(' '.join(map(str, best_solution)))
    
    print(f"Solution written to {output_file}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    
    return best_solution

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python mostly_clique_solver.py <graph_file> <output_file>")
        sys.exit(1)
    
    graph_file = sys.argv[1]
    output_file = sys.argv[2]
    
    solve_mostly_clique(graph_file, output_file)