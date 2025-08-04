import numpy as np
import scipy.sparse as sp
import networkx as nx
from collections import defaultdict
import time

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

def create_adjacency_matrix(G):
    """Create a sparse adjacency matrix from the graph."""
    print("Creating adjacency matrix...")
    start_time = time.time()
    
    # Create sparse adjacency matrix
    A = nx.to_scipy_sparse_array(G, format='csr')
    
    print(f"Adjacency matrix created in {time.time() - start_time:.2f} seconds")
    return A

def frank_wolfe_step(x, A, gamma):
    """Perform one step of the Frank-Wolfe algorithm."""
    # Compute the gradient of the objective function: A.x
    gradient = A.dot(x)
    
    # Find the vertex of the simplex that maximizes the inner product with negative gradient
    s = np.zeros_like(x)
    idx = np.argmax(gradient)
    s[idx] = 1.0
    
    # Update x using the Frank-Wolfe step
    x_new = (1 - gamma) * x + gamma * s
    
    return x_new, s

def frank_wolfe_mostly_clique(A, max_iter=1000, epsilon=1e-6):
    """
    Find a dense subgraph using the Frank-Wolfe algorithm.
    
    Parameters:
    A: Sparse adjacency matrix
    max_iter: Maximum number of iterations
    epsilon: Convergence threshold
    
    Returns:
    x: Relaxed solution vector
    """
    n = A.shape[0]
    
    # Initialize x uniformly over the simplex
    x = np.ones(n) / n
    
    print(f"Starting Frank-Wolfe algorithm for {max_iter} iterations...")
    start_time = time.time()
    
    for t in range(max_iter):
        # Compute the objective value: x^T A x (quadratic term)
        obj_val = x.T @ A @ x
        
        # Perform a Frank-Wolfe step with step size 2/(t+2)
        gamma = 2.0 / (t + 2)
        x_new, s = frank_wolfe_step(x, A, gamma)
        
        # Check convergence
        if np.linalg.norm(x_new - x) < epsilon:
            x = x_new
            break
        
        x = x_new
        
        if (t + 1) % 100 == 0:
            print(f"Iteration {t+1}, Objective: {obj_val:.4f}")
    
    print(f"Frank-Wolfe completed in {time.time() - start_time:.2f} seconds")
    print(f"Final objective value: {x.T @ A @ x:.4f}")
    
    return x

def extract_mostly_clique(x, A, threshold=0.1):
    """
    Extract a mostly-clique from the relaxed solution.
    
    Parameters:
    x: Relaxed solution vector
    A: Adjacency matrix
    threshold: Threshold for selecting nodes from the relaxed solution
    
    Returns:
    mostly_clique: List of node indices forming a mostly-clique
    """
    print(f"Extracting mostly-clique with threshold {threshold}...")
    
    # Select nodes with x[i] > threshold
    selected_nodes = np.where(x > threshold)[0]
    
    if len(selected_nodes) == 0:
        # If no nodes meet the threshold, take the top-k nodes by x value
        k = max(40, int(np.sqrt(A.shape[0])))  # Choose a reasonable k
        selected_nodes = np.argsort(x)[-k:]
    
    # Check if the selected nodes form a mostly-clique
    subgraph = A[selected_nodes, :][:, selected_nodes]
    edge_count = subgraph.sum() / 2  # Divide by 2 as edges are counted twice
    max_edges = len(selected_nodes) * (len(selected_nodes) - 1) / 2
    
    print(f"Selected {len(selected_nodes)} nodes with {edge_count} edges")
    print(f"Maximum possible edges: {max_edges}")
    print(f"Edge density: {edge_count / max_edges:.4f}")
    
    # If not a mostly-clique, use a greedy approach to extract one
    if edge_count <= max_edges / 2:
        print("Selected nodes don't form a mostly-clique. Using greedy extraction...")
        mostly_clique = greedy_extract_mostly_clique(selected_nodes, A)
    else:
        mostly_clique = selected_nodes.tolist()
    
    return mostly_clique

def greedy_extract_mostly_clique(candidate_nodes, A):
    """
    Greedily extract a mostly-clique from candidate nodes.
    
    Parameters:
    candidate_nodes: Array of node indices to consider
    A: Adjacency matrix
    
    Returns:
    mostly_clique: List of node indices forming a mostly-clique
    """
    # Start with an empty solution
    mostly_clique = []
    
    # Add nodes one by one
    for node in candidate_nodes:
        # Check if adding the node maintains the mostly-clique property
        if is_mostly_clique(mostly_clique + [node], A):
            mostly_clique.append(node)
    
    return mostly_clique

def is_mostly_clique(nodes, A):
    """
    Check if a set of nodes forms a mostly-clique.
    
    Parameters:
    nodes: List of node indices
    A: Adjacency matrix
    
    Returns:
    is_mc: Boolean indicating if nodes form a mostly-clique
    """
    if len(nodes) <= 1:
        return True
    
    subgraph = A[nodes, :][:, nodes]
    edge_count = subgraph.sum() / 2  # Divide by 2 as edges are counted twice
    max_edges = len(nodes) * (len(nodes) - 1) / 2
    
    return edge_count > max_edges / 2

def post_process_frank_wolfe(x, A, thresholds=[0.01, 0.05, 0.1, 0.2, 0.3]):
    """
    Try different thresholds for extracting mostly-cliques from the Frank-Wolfe solution.
    
    Parameters:
    x: Relaxed solution vector
    A: Adjacency matrix
    thresholds: List of thresholds to try
    
    Returns:
    best_mostly_clique: The largest mostly-clique found
    """
    best_mostly_clique = []
    
    for threshold in thresholds:
        mostly_clique = extract_mostly_clique(x, A, threshold)
        if len(mostly_clique) > len(best_mostly_clique):
            best_mostly_clique = mostly_clique
    
    return best_mostly_clique

def local_search(mostly_clique, A, max_iterations=100):
    """
    Perform local search to improve the mostly-clique.
    
    Parameters:
    mostly_clique: Initial mostly-clique
    A: Adjacency matrix
    max_iterations: Maximum number of iterations
    
    Returns:
    improved_clique: Improved mostly-clique
    """
    print("Performing local search...")
    start_time = time.time()
    
    current_clique = mostly_clique.copy()
    n = A.shape[0]
    
    # Set of nodes in the current clique
    clique_set = set(current_clique)
    
    for iter_num in range(max_iterations):
        improved = False
        
        # Try adding nodes
        for node in range(n):
            if node not in clique_set:
                test_clique = current_clique + [node]
                if is_mostly_clique(test_clique, A):
                    current_clique = test_clique
                    clique_set.add(node)
                    improved = True
        
        # Try removing nodes (only if it doesn't decrease the size)
        for node in list(clique_set):
            test_clique = [n for n in current_clique if n != node]
            
            # Check if we can remove this node and add two others
            # neighbors = set(np.nonzero(A[node])[0]) - clique_set
            # neighbors = set(A[node, :].nonzero()[1]) - clique_set
            # For a CSR sparse matrix, use these indices:

            row_start = A.indptr[node]
            row_end = A.indptr[node + 1]
            neighbors = set(A.indices[row_start:row_end]) - clique_set
            
            for neighbor1 in neighbors:
                for neighbor2 in neighbors:
                    if neighbor1 != neighbor2:
                        test_clique_2 = test_clique + [neighbor1, neighbor2]
                        if is_mostly_clique(test_clique_2, A) and len(test_clique_2) > len(current_clique):
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
    
    print(f"Local search completed in {time.time() - start_time:.2f} seconds")
    print(f"Mostly-clique size improved to: {len(current_clique)}")
    
    return current_clique

def save_solution(mostly_clique, reverse_mapping, output_file):
    """Save the solution to a file."""
    # Convert indices back to original node IDs
    original_nodes = [reverse_mapping[idx] for idx in mostly_clique]
    
    with open(output_file, 'w') as f:
        f.write(' '.join(map(str, original_nodes)))
    
    print(f"Solution saved to {output_file}")
    print(f"Mostly-clique size: {len(mostly_clique)}")

def verify_solution(mostly_clique, A):
    """Verify that the solution is indeed a mostly-clique."""
    if len(mostly_clique) <= 1:
        print("Warning: Solution contains only one node.")
        return True
    
    subgraph = A[mostly_clique, :][:, mostly_clique]
    edge_count = subgraph.sum() / 2  # Divide by 2 as edges are counted twice
    max_edges = len(mostly_clique) * (len(mostly_clique) - 1) / 2
    
    is_mc = edge_count > max_edges / 2
    
    print(f"Verification result:")
    print(f"- Nodes: {len(mostly_clique)}")
    print(f"- Edges: {edge_count}/{max_edges} ({edge_count/max_edges:.4f})")
    print(f"- Is mostly-clique: {is_mc}")
    
    return is_mc

def main():
    """Main function to run the mostly-clique finding algorithm."""
    input_file = "edges.csv"
    output_file = "mostly_clique_solution.txt"
    
    # Load the graph
    G, node_mapping, reverse_mapping = load_graph(input_file)
    
    # Preprocess the graph (remove low-degree nodes)
    G_filtered = preprocess_graph(G, min_degree=21)
    
    # Create adjacency matrix
    A = create_adjacency_matrix(G_filtered)
    
    # Run Frank-Wolfe algorithm
    x = frank_wolfe_mostly_clique(A, max_iter=1000)
    
    # Extract mostly-clique with different thresholds
    mostly_clique = post_process_frank_wolfe(x, A)
    
    # Improve solution with local search
    improved_clique = local_search(mostly_clique, A)
    
    # Verify the solution
    verify_solution(improved_clique, A)
    
    # Save the solution
    save_solution(improved_clique, reverse_mapping, output_file)

if __name__ == "__main__":
    main()