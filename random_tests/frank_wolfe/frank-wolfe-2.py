import numpy as np
import scipy.sparse as sp
import networkx as nx
from collections import defaultdict
import time
import random

def load_graph(file_path):
    """Load the graph from the edge list file."""
    print("Loading graph...")
    start_time = time.time()
    
    # Create a dictionary to store adjacency lists
    adjacency_dict = defaultdict(set)
    node_set = set()
    
    # Read edges from file
    with open(file_path, 'r') as f:
        for line in f:
            u, v = map(int, line.strip().split(','))
            adjacency_dict[u].add(v)
            adjacency_dict[v].add(u)
            node_set.add(u)
            node_set.add(v)
    
    # Create a mapping of node IDs to consecutive integers
    node_mapping = {node: idx for idx, node in enumerate(node_set)}
    reverse_mapping = {idx: node for node, idx in node_mapping.items()}
    
    # Create a networkx graph
    G = nx.Graph()
    
    # Add nodes with the new IDs
    G.add_nodes_from(range(len(node_set)))
    
    # Add edges with the new IDs
    for u in adjacency_dict:
        u_mapped = node_mapping[u]
        for v in adjacency_dict[u]:
            if u < v:  # Avoid adding edges twice
                v_mapped = node_mapping[v]
                G.add_edge(u_mapped, v_mapped)
    
    print(f"Graph loaded in {time.time() - start_time:.2f} seconds")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    return G, node_mapping, reverse_mapping

def preprocess_graph(G, min_degree=20):
    """Remove nodes with degree less than min_degree."""
    print(f"Preprocessing graph (removing nodes with degree < {min_degree})...")
    start_time = time.time()
    
    # Create a new graph with only high-degree nodes
    G_filtered = G.copy()
    
    # Iteratively remove nodes until no nodes with degree < min_degree remain
    while True:
        nodes_to_remove = [node for node, degree in G_filtered.degree() if degree < min_degree]
        if not nodes_to_remove:
            break
        G_filtered.remove_nodes_from(nodes_to_remove)
        print(f"Removed {len(nodes_to_remove)} nodes. Nodes remaining: {G_filtered.number_of_nodes()}")
    
    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
    print(f"Nodes after preprocessing: {G_filtered.number_of_nodes()}")
    print(f"Edges after preprocessing: {G_filtered.number_of_edges()}")
    
    return G_filtered

def find_initial_clique(G, min_size=3):
    """Find an initial clique to start with."""
    print(f"Finding initial clique of size at least {min_size}...")
    start_time = time.time()
    
    # Try to find cliques using NetworkX
    for clique in nx.find_cliques(G):
        if len(clique) >= min_size:
            print(f"Found initial clique of size {len(clique)} in {time.time() - start_time:.2f} seconds")
            return clique
    
    # If no clique of desired size is found, return the largest one
    largest_clique = max(nx.find_cliques(G), key=len)
    print(f"Found initial clique of size {len(largest_clique)} in {time.time() - start_time:.2f} seconds")
    return largest_clique

def is_mostly_clique(G, nodes):
    """Check if a set of nodes forms a mostly-clique."""
    if len(nodes) <= 1:
        return True
    
    subgraph = G.subgraph(nodes)
    edge_count = subgraph.number_of_edges()
    max_edges = len(nodes) * (len(nodes) - 1) / 2
    
    return edge_count > max_edges / 2

def count_connections(G, node, current_set):
    """Count the number of connections from a node to a set of nodes."""
    return sum(1 for n in current_set if G.has_edge(node, n))

def greedy_expand(G, initial_set, candidate_nodes=None):
    """
    Greedily expand an initial set to form a larger mostly-clique.
    
    Parameters:
    G: NetworkX graph
    initial_set: Initial set of nodes to expand from
    candidate_nodes: Set of candidate nodes to consider adding (default: all nodes)
    
    Returns:
    expanded_set: Expanded mostly-clique
    """
    print("Greedily expanding the initial set...")
    start_time = time.time()
    
    current_set = list(initial_set)
    if candidate_nodes is None:
        candidate_nodes = set(G.nodes()) - set(current_set)
    else:
        candidate_nodes = set(candidate_nodes) - set(current_set)
    
    # Sort nodes by degree (highest first) for initial ordering
    candidate_nodes = sorted(candidate_nodes, key=lambda n: G.degree(n), reverse=True)
    
    # Keep track of edge counts for efficiency
    current_edges = G.subgraph(current_set).number_of_edges()
    current_nodes = len(current_set)
    
    added = True
    while added and candidate_nodes:
        added = False
        
        # Calculate the minimum number of connections needed to maintain mostly-clique
        min_conn_needed = []
        for node in candidate_nodes:
            new_nodes = current_nodes + 1
            new_max_edges = new_nodes * (new_nodes - 1) / 2
            min_conn = int(new_max_edges / 2) + 1 - current_edges
            min_conn_needed.append((node, min_conn))
        
        # Sort by minimum connections needed (ascending)
        min_conn_needed.sort(key=lambda x: x[1])
        
        for node, min_conn in min_conn_needed:
            connections = count_connections(G, node, current_set)
            
            if connections >= min_conn:
                current_set.append(node)
                current_edges += connections
                current_nodes += 1
                candidate_nodes.remove(node)
                added = True
                break
    
    print(f"Expansion completed in {time.time() - start_time:.2f} seconds")
    print(f"Expanded to size {len(current_set)}")
    
    return current_set

def k_core_decomposition(G, min_k=20):
    """
    Perform k-core decomposition to find dense subgraphs.
    
    Parameters:
    G: NetworkX graph
    min_k: Minimum k-core to extract
    
    Returns:
    cores: Dictionary mapping k to the k-core subgraph
    """
    print(f"Performing k-core decomposition...")
    start_time = time.time()
    
    # Compute the k-core decomposition
    core_numbers = nx.core_number(G)
    max_core = max(core_numbers.values())
    
    print(f"Maximum core number: {max_core}")
    
    # Extract cores of interest
    cores = {}
    for k in range(min_k, max_core + 1):
        core_nodes = [n for n, c in core_numbers.items() if c >= k]
        if core_nodes:
            cores[k] = G.subgraph(core_nodes)
    
    print(f"K-core decomposition completed in {time.time() - start_time:.2f} seconds")
    print(f"Found {len(cores)} cores with k >= {min_k}")
    
    return cores

def randomized_greedy(G, num_iterations=10, sample_size=None):
    """
    Perform multiple runs of randomized greedy expansion.
    
    Parameters:
    G: NetworkX graph
    num_iterations: Number of randomized runs
    sample_size: Number of nodes to sample for each run (default: 10% of nodes)
    
    Returns:
    best_mostly_clique: The largest mostly-clique found
    """
    print(f"Performing {num_iterations} randomized greedy expansions...")
    start_time = time.time()
    
    n = G.number_of_nodes()
    if sample_size is None:
        sample_size = max(100, int(0.1 * n))
    
    best_mostly_clique = []
    nodes = list(G.nodes())
    
    for i in range(num_iterations):
        # Sample nodes with preference for high-degree nodes
        node_degrees = [(node, G.degree(node)) for node in nodes]
        node_degrees.sort(key=lambda x: x[1], reverse=True)
        
        # Take top nodes plus some random ones
        top_nodes = [node for node, _ in node_degrees[:sample_size//2]]
        random_nodes = random.sample([node for node, _ in node_degrees[sample_size//2:]], 
                                     min(sample_size//2, len(node_degrees) - sample_size//2))
        sample_nodes = top_nodes + random_nodes
        
        # Find an initial clique in the sample
        sample_G = G.subgraph(sample_nodes)
        initial_clique = find_initial_clique(sample_G, min_size=3)
        
        # Expand to a mostly-clique
        mostly_clique = greedy_expand(G, initial_clique)
        
        if len(mostly_clique) > len(best_mostly_clique):
            best_mostly_clique = mostly_clique
            print(f"New best mostly-clique found: size {len(best_mostly_clique)}")
        
        print(f"Iteration {i+1}/{num_iterations}: Found mostly-clique of size {len(mostly_clique)}")
    
    print(f"Randomized greedy completed in {time.time() - start_time:.2f} seconds")
    print(f"Best mostly-clique size: {len(best_mostly_clique)}")
    
    return best_mostly_clique

def local_search(G, mostly_clique, max_iterations=100):
    """
    Perform local search to improve the mostly-clique.
    
    Parameters:
    G: NetworkX graph
    mostly_clique: Initial mostly-clique
    max_iterations: Maximum number of iterations
    
    Returns:
    improved_clique: Improved mostly-clique
    """
    print("Performing local search...")
    start_time = time.time()
    
    current_clique = list(mostly_clique)
    
    # Set of nodes in the current clique
    clique_set = set(current_clique)
    
    for iter_num in range(max_iterations):
        improved = False
        
        # Try adding nodes
        candidates = []
        for node in G.nodes():
            if node not in clique_set:
                connections = count_connections(G, node, current_clique)
                candidates.append((node, connections))
        
        # Sort candidates by number of connections (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        for node, _ in candidates:
            test_clique = current_clique + [node]
            if is_mostly_clique(G, test_clique):
                current_clique = test_clique
                clique_set.add(node)
                improved = True
                break
        
        # Try removing nodes (only if it doesn't decrease the size)
        for node in list(clique_set):
            test_clique = [n for n in current_clique if n != node]
            
            # Check if we can remove this node and add two others
            neighbors = [n for n in G.neighbors(node) if n not in clique_set]
            
            for i, neighbor1 in enumerate(neighbors):
                for neighbor2 in neighbors[i+1:]:
                    test_clique_2 = test_clique + [neighbor1, neighbor2]
                    if is_mostly_clique(G, test_clique_2) and len(test_clique_2) > len(current_clique):
                        current_clique = test_clique_2
                        clique_set.remove(node)
                        clique_set.add(neighbor1)
                        clique_set.add(neighbor2)
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
            
        if (iter_num + 1) % 10 == 0:
            print(f"Iteration {iter_num+1}: Mostly-clique size = {len(current_clique)}")
    
    print(f"Local search completed in {time.time() - start_time:.2f} seconds")
    print(f"Mostly-clique size improved to: {len(current_clique)}")
    
    return current_clique

def verify_solution(G, mostly_clique):
    """Verify that the solution is indeed a mostly-clique."""
    if len(mostly_clique) <= 1:
        print("Warning: Solution contains only one node.")
        return True
    
    subgraph = G.subgraph(mostly_clique)
    edge_count = subgraph.number_of_edges()
    max_edges = len(mostly_clique) * (len(mostly_clique) - 1) / 2
    
    is_mc = edge_count > max_edges / 2
    
    print(f"Verification result:")
    print(f"- Nodes: {len(mostly_clique)}")
    print(f"- Edges: {edge_count}/{max_edges} ({edge_count/max_edges:.4f})")
    print(f"- Is mostly-clique: {is_mc}")
    
    return is_mc

def save_solution(mostly_clique, reverse_mapping, output_file):
    """Save the solution to a file."""
    # Convert indices back to original node IDs
    original_nodes = [reverse_mapping[idx] for idx in mostly_clique]
    
    with open(output_file, 'w') as f:
        f.write(' '.join(map(str, original_nodes)))
    
    print(f"Solution saved to {output_file}")
    print(f"Mostly-clique size: {len(mostly_clique)}")

def main():
    """Main function to find a large mostly-clique."""
    input_file = "edges.csv"
    output_file = "mostly_clique_solution.txt"
    
    # Load the graph
    G, node_mapping, reverse_mapping = load_graph(input_file)
    
    # Preprocess the graph (remove low-degree nodes)
    G_filtered = preprocess_graph(G, min_degree=21)
    
    # Approach 1: K-core decomposition
    cores = k_core_decomposition(G_filtered, min_k=30)
    
    best_mostly_clique = []
    
    # For each core, try to find a mostly-clique
    for k, core in cores.items():
        if len(core) > 1000:  # Skip very large cores
            continue
            
        print(f"Processing {k}-core with {core.number_of_nodes()} nodes...")
        
        # Find initial clique
        initial_clique = find_initial_clique(core, min_size=3)
        
        # Expand to a mostly-clique
        mostly_clique = greedy_expand(G_filtered, initial_clique)
        
        if len(mostly_clique) > len(best_mostly_clique):
            best_mostly_clique = mostly_clique
            print(f"New best mostly-clique found: size {len(best_mostly_clique)}")
    
    # Approach 2: Randomized greedy
    # randomized_mostly_clique = randomized_greedy(G_filtered, num_iterations=5)
    
    # if len(randomized_mostly_clique) > len(best_mostly_clique):
    #     best_mostly_clique = randomized_mostly_clique
    
    # # Apply local search to the best solution
    # if best_mostly_clique:
    #     improved_clique = local_search(G_filtered, best_mostly_clique)
        
    #     # Verify the solution
    #     verify_solution(G_filtered, improved_clique)
        
    #     # Save the solution
    #     save_solution(improved_clique, reverse_mapping, output_file)
    # else:
    #     print("Failed to find a mostly-clique.")

    verify_solution(G_filtered, best_mostly_clique)
    save_solution(best_mostly_clique, reverse_mapping, output_file)

if __name__ == "__main__":
    main()