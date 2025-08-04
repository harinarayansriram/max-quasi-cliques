import networkx as nx
import numpy as np
import pandas as pd
import random
import time
import os
from collections import defaultdict
import argparse
from tqdm import tqdm

def load_graph(filepath):
    """Load the graph from the edge list file."""
    print(f"Loading graph from {filepath}...")
    edges = pd.read_csv(filepath, header=None, names=['source', 'target'])
    
    # Convert to NetworkX graph
    G = nx.Graph()
    G.add_edges_from(edges.values)
    
    print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def is_mostly_clique(G, nodes):
    """Check if the subgraph induced by nodes is a mostly-clique."""
    subgraph = G.subgraph(nodes)
    edge_count = subgraph.number_of_edges()
    node_count = len(nodes)
    max_possible_edges = (node_count * (node_count - 1)) // 2
    return edge_count > max_possible_edges // 2

def get_edge_density(G, nodes):
    """Calculate the edge density of the subgraph induced by nodes."""
    if len(nodes) <= 1:
        return 0.0
    
    subgraph = G.subgraph(nodes)
    edge_count = subgraph.number_of_edges()
    node_count = len(nodes)
    max_possible_edges = (node_count * (node_count - 1)) // 2
    
    return edge_count / max_possible_edges if max_possible_edges > 0 else 0.0

def k_core_decomposition(G, min_k=10):
    """Extract k-cores of the graph."""
    print("Performing k-core decomposition...")
    core_numbers = nx.core_number(G)
    max_core = max(core_numbers.values())
    
    print(f"Maximum core number: {max_core}")
    
    # Start with cores that are at least min_k
    k = max(min_k, max_core // 2)
    while k <= max_core:
        core_nodes = [node for node, c in core_numbers.items() if c >= k]
        if len(core_nodes) < 1000:  # If core is small enough to process
            print(f"Found {k}-core with {len(core_nodes)} nodes")
            return core_nodes
        k += 1
    
    # If all cores are too large, return the highest one with a size limit
    core_nodes = [node for node, c in core_numbers.items() if c >= max_core]
    print(f"Using {max_core}-core with {len(core_nodes)} nodes")
    if len(core_nodes) > 5000:
        print(f"Limiting to 5000 random nodes from {max_core}-core")
        return random.sample(core_nodes, 5000)
    return core_nodes

def greedy_mostly_clique(G, seed_nodes=None, max_iterations=1000):
    """Greedy algorithm to find a large mostly-clique."""
    if seed_nodes is None:
        # Start with a random node
        seed_nodes = [random.choice(list(G.nodes()))]
    
    current_solution = list(seed_nodes)
    best_solution = list(current_solution)
    best_size = len(best_solution)
    
    print(f"Starting greedy search with {len(current_solution)} seed nodes...")
    
    # Get set of candidate nodes (neighbors of the seed nodes)
    candidates = set()
    for node in current_solution:
        candidates.update(G.neighbors(node))
    candidates = candidates - set(current_solution)
    
    pbar = tqdm(total=max_iterations)
    
    iteration = 0
    while candidates and iteration < max_iterations:
        # Find the node that adds the most edges to the current solution
        best_node = None
        best_density = get_edge_density(G, current_solution)
        
        # Sample a subset of candidates if there are too many
        sample_candidates = list(candidates)
        if len(sample_candidates) > 1000:
            sample_candidates = random.sample(sample_candidates, 1000)
        
        for node in sample_candidates:
            temp_solution = current_solution + [node]
            density = get_edge_density(G, temp_solution)
            
            if density > best_density:
                best_density = density
                best_node = node
        
        if best_node:
            current_solution.append(best_node)
            candidates.remove(best_node)
            
            # Update candidate set with neighbors
            for neighbor in G.neighbors(best_node):
                if neighbor not in current_solution:
                    candidates.add(neighbor)
            
            # Check if it's still a mostly-clique
            if is_mostly_clique(G, current_solution):
                if len(current_solution) > best_size:
                    best_solution = list(current_solution)
                    best_size = len(best_solution)
                    print(f"Found larger mostly-clique with {best_size} nodes (density: {best_density:.4f})")
                    save_solution(best_solution, "current_best_solution.txt")
            else:
                # If not a mostly-clique, remove the node
                current_solution.pop()
        else:
            # If no node improves the solution, break
            break
        
        iteration += 1
        pbar.update(1)
    
    pbar.close()
    return best_solution

def local_search(G, initial_solution, max_iterations=500):
    """Local search to improve a mostly-clique."""
    current_solution = list(initial_solution)
    best_solution = list(current_solution)
    best_density = get_edge_density(G, best_solution)
    
    print(f"Starting local search from solution with {len(current_solution)} nodes...")
    
    # Potential neighbors to consider adding
    candidates = set()
    for node in current_solution:
        candidates.update(G.neighbors(node))
    candidates = candidates - set(current_solution)
    
    pbar = tqdm(total=max_iterations)
    
    for iteration in range(max_iterations):
        improved = False
        
        # Try adding a node
        if candidates:
            sample_candidates = list(candidates)
            if len(sample_candidates) > 500:
                sample_candidates = random.sample(sample_candidates, 500)
                
            for node in sample_candidates:
                temp_solution = current_solution + [node]
                if is_mostly_clique(G, temp_solution):
                    temp_density = get_edge_density(G, temp_solution)
                    if temp_density >= best_density and len(temp_solution) > len(best_solution):
                        current_solution = temp_solution
                        best_solution = temp_solution
                        best_density = temp_density
                        improved = True
                        print(f"Improved to {len(best_solution)} nodes (density: {best_density:.4f})")
                        save_solution(best_solution, "current_best_solution.txt")
                        
                        # Update candidates
                        candidates.add(node)
                        for n in G.neighbors(node):
                            if n not in current_solution:
                                candidates.add(n)
                        candidates.remove(node)
                        break
        
        # Try removing a node and adding two
        if not improved and len(current_solution) > 5:
            # Try removing a node with the lowest degree in the induced subgraph
            subgraph = G.subgraph(current_solution)
            node_degrees = dict(subgraph.degree())
            nodes_by_degree = sorted(node_degrees.items(), key=lambda x: x[1])
            
            # Try removing some of the lowest degree nodes
            for node, _ in nodes_by_degree[:min(10, len(nodes_by_degree))]:
                temp_solution = [n for n in current_solution if n != node]
                
                # Get potential nodes to add
                potential_adds = set()
                for n in temp_solution:
                    potential_adds.update(G.neighbors(n))
                potential_adds = potential_adds - set(temp_solution)
                
                if len(potential_adds) >= 2:
                    # Try adding pairs of nodes
                    sample_adds = list(potential_adds)
                    if len(sample_adds) > 50:
                        sample_adds = random.sample(sample_adds, 50)
                    
                    for i in range(len(sample_adds)):
                        for j in range(i+1, len(sample_adds)):
                            add1, add2 = sample_adds[i], sample_adds[j]
                            new_solution = temp_solution + [add1, add2]
                            
                            if is_mostly_clique(G, new_solution):
                                new_density = get_edge_density(G, new_solution)
                                if new_density >= best_density and len(new_solution) > len(best_solution):
                                    current_solution = new_solution
                                    best_solution = new_solution
                                    best_density = new_density
                                    improved = True
                                    print(f"Swap improved to {len(best_solution)} nodes (density: {best_density:.4f})")
                                    save_solution(best_solution, "current_best_solution.txt")
                                    
                                    # Update candidates
                                    candidates.add(add1)
                                    candidates.add(add2)
                                    for n in G.neighbors(add1):
                                        if n not in current_solution:
                                            candidates.add(n)
                                    for n in G.neighbors(add2):
                                        if n not in current_solution:
                                            candidates.add(n)
                                    
                                    break
                        if improved:
                            break
                if improved:
                    break
        
        if not improved:
            # If no improvement was made, try a random restart
            if random.random() < 0.2:  # 20% chance of random restart
                print("Performing random restart...")
                # Keep some percentage of the current solution
                keep_percentage = 0.7
                keep_count = int(len(current_solution) * keep_percentage)
                current_solution = random.sample(current_solution, keep_count)
                
                # Update candidates
                candidates = set()
                for node in current_solution:
                    candidates.update(G.neighbors(node))
                candidates = candidates - set(current_solution)
        
        pbar.update(1)
    
    pbar.close()
    return best_solution

def save_solution(solution, filename):
    """Save the current solution to a file."""
    with open(filename, 'w') as f:
        f.write(' '.join(map(str, solution)))
    print(f"Saved solution with {len(solution)} nodes to {filename}")

def multi_start_search(G, k_core_nodes, num_starts=5, iterations_per_start=200):
    """Run multiple greedy searches with different starting points."""
    best_solution = []
    
    print(f"Running {num_starts} multi-start searches...")
    
    for i in range(num_starts):
        print(f"\nStart {i+1}/{num_starts}")
        
        # Select a random starting point from the k-core
        if len(k_core_nodes) > 0:
            seed_nodes = [random.choice(k_core_nodes)]
            
            # Add some neighbors with high core numbers
            neighbors = list(G.neighbors(seed_nodes[0]))
            if neighbors:
                seed_nodes.extend(random.sample(neighbors, min(10, len(neighbors))))
        else:
            # If no k-core was found, start with a random node
            seed_nodes = [random.choice(list(G.nodes()))]
        
        # Run greedy algorithm
        solution = greedy_mostly_clique(G, seed_nodes, max_iterations=iterations_per_start)
        
        # Run local search to improve the solution
        solution = local_search(G, solution, max_iterations=iterations_per_start*2)
        
        if len(solution) > len(best_solution):
            best_solution = solution
            save_solution(best_solution, "best_solution.txt")
    
    return best_solution

def main():
    parser = argparse.ArgumentParser(description='Find a large mostly-clique in a graph.')
    parser.add_argument('--input', type=str, default='edges.csv', help='Input edge list file')
    parser.add_argument('--output', type=str, default='solution.txt', help='Output solution file')
    parser.add_argument('--starts', type=int, default=5, help='Number of multi-starts')
    parser.add_argument('--iterations', type=int, default=200, help='Iterations per start')
    parser.add_argument('--min-k', type=int, default=10, help='Minimum k for k-core decomposition')
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Load the graph
    G = load_graph(args.input)
    
    # Find a k-core to use as a starting point
    k_core_nodes = k_core_decomposition(G, min_k=args.min_k)
    
    # Run multi-start search
    best_solution = multi_start_search(G, k_core_nodes, num_starts=args.starts, iterations_per_start=args.iterations)
    
    # Save the final solution
    save_solution(best_solution, args.output)
    
    elapsed_time = time.time() - start_time
    print(f"\nBest solution found: {len(best_solution)} nodes")
    print(f"Total runtime: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()