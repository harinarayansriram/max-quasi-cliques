import networkx as nx
import pandas as pd
import numpy as np
import time
import random
from collections import Counter

def load_graph(edge_file):
    """Load graph from edge list file."""
    print("Loading graph...")
    df = pd.read_csv(edge_file, header=None, names=['source', 'target'])
    G = nx.Graph()
    G.add_edges_from(zip(df['source'], df['target']))
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def is_quasi_clique(G, nodes):
    """Check if the subgraph induced by nodes is a quasi-clique."""
    if len(nodes) <= 1:
        return False, 0
    
    subgraph = G.subgraph(nodes)
    possible_edges = len(nodes) * (len(nodes) - 1) / 2
    actual_edges = subgraph.number_of_edges()
    density = actual_edges / possible_edges
    return actual_edges > possible_edges / 2, density

def evaluate_node_for_addition(G, current_nodes, candidate):
    """Evaluate how good a candidate node is for addition to the quasi-clique."""
    connections = sum(1 for n in current_nodes if G.has_edge(candidate, n))
    return connections / len(current_nodes) if current_nodes else 0

def node_removal_benefit(G, current_nodes, node):
    """Calculate the benefit of removing a node from the quasi-clique."""
    connections = sum(1 for n in current_nodes if G.has_edge(node, n) and n != node)
    return 1 - (connections / (len(current_nodes) - 1)) if len(current_nodes) > 1 else 0

def find_initial_quasi_clique(G, seed_size=50, expansion_iterations=100):
    """Find an initial quasi-clique using a greedy approach."""
    print("Finding initial quasi-clique...")
    
    # Start with high-degree nodes
    nodes_by_degree = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    seed_nodes = [n for n, _ in nodes_by_degree[:seed_size]]
    
    # Create a subgraph with these nodes
    current_nodes = set(seed_nodes)
    
    # Iteratively improve by removing poor connections and adding good ones
    for _ in range(expansion_iterations):
        # Remove nodes with poor connectivity
        nodes_to_remove = []
        for node in current_nodes:
            connections = sum(1 for n in current_nodes if G.has_edge(node, n) and n != node)
            if connections < len(current_nodes) / 2:
                nodes_to_remove.append(node)
                
        for node in nodes_to_remove:
            current_nodes.remove(node)
            
        if not current_nodes:
            print("Lost all nodes during pruning. Restarting with different seed...")
            return find_initial_quasi_clique(G, seed_size=seed_size//2, expansion_iterations=expansion_iterations)
        
        # Find candidate nodes with good connectivity to current set
        candidates = set()
        for node in current_nodes:
            candidates.update(G.neighbors(node))
        candidates -= current_nodes
        
        # Score candidates
        candidate_scores = [(n, evaluate_node_for_addition(G, current_nodes, n)) 
                           for n in candidates]
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Add best candidates
        for node, score in candidate_scores:
            if score > 0.5:  # Must connect to at least half of current nodes
                current_nodes.add(node)
                
        # Check if we have a quasi-clique
        is_valid, density = is_quasi_clique(G, current_nodes)
        if is_valid:
            print(f"Found initial quasi-clique with {len(current_nodes)} nodes, density: {density:.4f}")
            return list(current_nodes)
    
    print("Could not find initial quasi-clique. Trying with different parameters...")
    return find_initial_quasi_clique(G, seed_size=seed_size//2, expansion_iterations=expansion_iterations*2)

def grow_quasi_clique(G, initial_nodes, max_iterations=500):
    """Grow a quasi-clique using a greedy algorithm."""
    print(f"Growing quasi-clique from {len(initial_nodes)} nodes...")
    
    current_nodes = set(initial_nodes)
    best_size = len(current_nodes)
    best_nodes = list(current_nodes)
    best_density = 0
    
    is_valid, best_density = is_quasi_clique(G, current_nodes)
    if not is_valid:
        print("Initial nodes do not form a quasi-clique. Fixing...")
        # Remove nodes until we have a valid quasi-clique
        nodes_to_remove = []
        for node in current_nodes:
            temp_nodes = current_nodes - {node}
            is_valid, _ = is_quasi_clique(G, temp_nodes)
            if is_valid:
                nodes_to_remove.append(node)
        
        for node in nodes_to_remove:
            current_nodes.remove(node)
            is_valid, best_density = is_quasi_clique(G, current_nodes)
            if is_valid:
                break
        
        best_nodes = list(current_nodes)
        best_size = len(best_nodes)
    
    # Keep track of iterations without improvement
    stagnant_iterations = 0
    
    for iteration in range(max_iterations):
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Current size = {len(current_nodes)}, Best size = {best_size}")
        
        # Get candidate nodes for addition
        candidates = set()
        for node in current_nodes:
            candidates.update(G.neighbors(node))
        candidates -= current_nodes
        
        # Score candidates
        candidate_scores = []
        for node in candidates:
            connections = sum(1 for n in current_nodes if G.has_edge(node, n))
            score = connections / len(current_nodes)
            candidate_scores.append((node, score))
        
        # Sort candidates by score
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Try adding the best candidates
        improved = False
        for node, score in candidate_scores[:50]:  # Consider top 50 candidates
            temp_nodes = current_nodes | {node}
            is_valid, density = is_quasi_clique(G, temp_nodes)
            
            if is_valid:
                current_nodes = temp_nodes
                improved = True
                
                if len(current_nodes) > best_size:
                    best_size = len(current_nodes)
                    best_nodes = list(current_nodes)
                    best_density = density
                    print(f"New best: {best_size} nodes with density {best_density:.4f}")
                    stagnant_iterations = 0
                break
        
        # If we couldn't add any nodes, try removing some bad nodes
        if not improved:
            # Score nodes for removal
            removal_scores = []
            for node in current_nodes:
                connections = sum(1 for n in current_nodes if G.has_edge(node, n) and n != node)
                score = connections / (len(current_nodes) - 1)
                removal_scores.append((node, score))
            
            # Remove some of the worst-connected nodes
            removal_scores.sort(key=lambda x: x[1])
            for node, _ in removal_scores[:5]:  # Try removing up to 5 bad nodes
                temp_nodes = current_nodes - {node}
                is_valid, _ = is_quasi_clique(G, temp_nodes)
                
                if is_valid:
                    current_nodes = temp_nodes
                    improved = True
                    break
        
        # If we still didn't improve, try a more radical move
        if not improved:
            stagnant_iterations += 1
            
            # After too many stagnant iterations, restart from best solution with some perturbation
            if stagnant_iterations > 50:
                print("Stagnated. Restarting from best solution with perturbation...")
                stagnant_iterations = 0
                current_nodes = set(best_nodes)
                
                # Remove some random nodes
                nodes_to_remove = random.sample(list(current_nodes), min(10, len(current_nodes)//10))
                for node in nodes_to_remove:
                    current_nodes.remove(node)
                
                # Add some random neighbors
                candidates = set()
                for node in current_nodes:
                    candidates.update(G.neighbors(node))
                candidates -= current_nodes
                
                if candidates:
                    nodes_to_add = random.sample(list(candidates), min(5, len(candidates)))
                    for node in nodes_to_add:
                        temp_nodes = current_nodes | {node}
                        is_valid, _ = is_quasi_clique(G, temp_nodes)
                        if is_valid:
                            current_nodes = temp_nodes
    
    return best_nodes, best_density

def extract_quasi_clique_from_known(G, known_size=175):
    """Try to find a large quasi-clique by starting from high-degree nodes."""
    print(f"Extracting quasi-clique using high-degree nodes...")
    
    # Get high-degree nodes
    nodes_by_degree = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    top_nodes = [n for n, _ in nodes_by_degree[:known_size*2]]
    
    # Create a subgraph with these nodes
    subgraph = G.subgraph(top_nodes)
    
    # Find k-core with k high enough to ensure density > 0.5
    k = int(known_size * 0.5)  # Start with a reasonable k value
    core_nodes = list(nx.k_core(subgraph, k=k).nodes())
    
    while len(core_nodes) > known_size * 2 and k < known_size:
        k += 5
        core_subgraph = nx.k_core(subgraph, k=k)
        if not core_subgraph:
            k -= 5
            break
        core_nodes = list(core_subgraph.nodes())
    
    print(f"Found {k}-core with {len(core_nodes)} nodes")
    
    # If too many nodes, take a subset
    if len(core_nodes) > known_size * 1.5:
        # Take nodes with highest degree within the core
        core_degrees = subgraph.subgraph(core_nodes).degree()
        core_nodes = [n for n, _ in sorted(core_degrees, key=lambda x: x[1], reverse=True)[:known_size]]
    
    # Check if we have a quasi-clique
    is_valid, density = is_quasi_clique(G, core_nodes)
    
    if is_valid:
        print(f"Found initial quasi-clique with {len(core_nodes)} nodes, density: {density:.4f}")
        return core_nodes
    else:
        print(f"Initial nodes not a quasi-clique (density: {density:.4f}). Running greedy algorithm...")
        
        # Try finding largest subset that forms a quasi-clique
        current_nodes = set(core_nodes)
        while current_nodes and not is_valid:
            # Remove node with lowest connectivity
            node_connections = {}
            for node in current_nodes:
                connections = sum(1 for n in current_nodes if G.has_edge(node, n) and n != node)
                node_connections[node] = connections / (len(current_nodes) - 1)
            
            worst_node = min(node_connections, key=node_connections.get)
            current_nodes.remove(worst_node)
            
            is_valid, density = is_quasi_clique(G, current_nodes)
        
        if current_nodes:
            print(f"Found valid quasi-clique with {len(current_nodes)} nodes, density: {density:.4f}")
            return list(current_nodes)
        else:
            print("Could not find valid quasi-clique from k-core. Trying greedy approach...")
            return find_initial_quasi_clique(G)

def local_search(G, initial_nodes, iterations=500):
    """Perform local search to improve quasi-clique size."""
    print(f"Running local search from {len(initial_nodes)} nodes...")
    
    current_nodes = set(initial_nodes)
    best_nodes = list(current_nodes)
    best_size = len(best_nodes)
    
    neighbors_cache = {}
    
    for iteration in range(iterations):
        if iteration % 10 == 0:
            print(f"Local search iteration {iteration}: current best = {best_size}")
            
        # Get all neighboring nodes
        neighbors = set()
        for node in current_nodes:
            if node not in neighbors_cache:
                neighbors_cache[node] = set(G.neighbors(node))
            neighbors.update(neighbors_cache[node])
        neighbors -= current_nodes
        
        # Try adding a node
        added = False
        for node in sorted(neighbors, key=lambda n: sum(1 for v in current_nodes if G.has_edge(n, v)), reverse=True):
            temp_nodes = current_nodes | {node}
            is_valid, _ = is_quasi_clique(G, temp_nodes)
            if is_valid:
                current_nodes = temp_nodes
                added = True
                
                if len(current_nodes) > best_size:
                    best_size = len(current_nodes)
                    best_nodes = list(current_nodes)
                    print(f"New best: {best_size} nodes")
                break
        
        # If no node could be added, try swapping
        if not added:
            # Calculate connections for each node in current solution
            internal_connections = {}
            for node in current_nodes:
                internal_connections[node] = sum(1 for n in current_nodes if G.has_edge(node, n) and n != node)
            
            # Find nodes with lowest internal connections
            worst_nodes = sorted(current_nodes, key=lambda n: internal_connections[n])[:10]
            
            # Try swapping with external nodes
            swapped = False
            for bad_node in worst_nodes:
                for ext_node in neighbors:
                    # Skip if already connected to too few nodes
                    ext_connections = sum(1 for n in current_nodes if G.has_edge(ext_node, n) and n != bad_node)
                    if ext_connections <= internal_connections[bad_node]:
                        continue
                        
                    # Try the swap
                    temp_nodes = (current_nodes - {bad_node}) | {ext_node}
                    is_valid, _ = is_quasi_clique(G, temp_nodes)
                    
                    if is_valid:
                        current_nodes = temp_nodes
                        swapped = True
                        break
                
                if swapped:
                    break
            
            # If no improvement after swapping, apply more aggressive perturbation
            if not swapped and iteration % 50 == 49:
                print("No improvement. Applying perturbation...")
                
                # Remove some of the worst nodes
                worst_nodes = sorted(current_nodes, 
                                     key=lambda n: sum(1 for v in current_nodes if G.has_edge(n, v) and v != n))[:5]
                
                for node in worst_nodes:
                    current_nodes.remove(node)