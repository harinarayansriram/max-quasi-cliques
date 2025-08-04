import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.optimize import minimize
import cvxpy as cp
import time
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

def compute_density(G, nodes):
    """Compute density of the subgraph induced by nodes."""
    if len(nodes) <= 1:
        return 1.0
        
    subgraph = G.subgraph(nodes)
    n = len(nodes)
    m = subgraph.number_of_edges()
    max_edges = n * (n - 1) / 2
    
    return m / max_edges

def is_mostly_clique(G, nodes):
    """Check if the subgraph induced by nodes is a mostly-clique."""
    return compute_density(G, nodes) > 0.5

def greedy_expansion(G, seed_nodes, max_iterations=1000):
    """Greedily expand the seed set while maintaining the mostly-clique property."""
    current = set(seed_nodes)
    candidates = set(G.nodes()) - current
    
    improved = True
    iterations = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Score candidates by connections to current set
        scores = {}
        for node in candidates:
            scores[node] = sum(1 for n in current if G.has_edge(node, n))
        
        # Try adding nodes in order of decreasing score
        for node in sorted(scores.keys(), key=lambda n: scores[n], reverse=True):
            test_set = current | {node}
            if is_mostly_clique(G, test_set):
                print('Iteration {}: Added node {} with score {}; size is now {}'.format(iterations, node, scores[node], len(test_set)))
                current = test_set
                candidates.remove(node)
                improved = True
                break
    
    return list(current)

def relaxed_mostly_clique_cvxpy(G, initial_size=200, max_iter=10):
    """Solve a relaxed version of the mostly-clique problem using CVXPY."""
    print("Setting up CVXPY optimization...")
    
    # Get adjacency matrix and convert to sparse format
    A = nx.to_scipy_sparse_array(G)
    n = G.number_of_nodes()
    
    # Create variables and constraints
    x = cp.Variable(n)
    
    # Objective: maximize sum of x
    objective = cp.Maximize(cp.sum(x))
    
    # Constraints:
    # 1. 0 <= x <= 1 for all i
    # 2. Sum of edges >= 0.5 * sum of all possible edges
    constraints = [
        0 <= x, x <= 1,
    ]
    
    # Create the density constraint
    # For each pair (i,j), the contribution is:
    # - if (i,j) is an edge: x_i * x_j
    # - if (i,j) is not an edge: -0.5 * x_i * x_j
    
    # This can be expressed as x^T (A - 0.5 * J) x > 0
    # where J is all-ones matrix with zeros on diagonal
    
    # We'll use a different approach: limit to a subset of nodes for tractability
    largest_components = list(nx.connected_components(G))
    largest_components.sort(key=len, reverse=True)
    
    component_nodes = list(largest_components[0])[:initial_size]
    component_size = len(component_nodes)
    
    # Create subgraph
    subgraph = G.subgraph(component_nodes)
    sub_adj = nx.to_numpy_array(subgraph)
    
    # Node mapping
    node_mapping = {node: i for i, node in enumerate(component_nodes)}
    inv_mapping = {i: node for node, i in node_mapping.items()}
    
    # Create CVXPY problem for the subgraph
    y = cp.Variable(component_size)
    sub_objective = cp.Maximize(cp.sum(y))
    
    # Create constraint matrix: A - 0.5 * (J - I)
    # where J is all-ones matrix and I is identity
    J = np.ones((component_size, component_size))
    np.fill_diagonal(J, 0)  # zero out diagonal
    constraint_matrix = sub_adj - 0.5 * J
    
    # The constraint: y^T * constraint_matrix * y >= 0
    # For CVXPY, it's cleaner to use the built-in quad_form
    sub_constraints = [
        0 <= y, y <= 1,
        cp.quad_form(y, constraint_matrix) >= 0
    ]
    
    sub_problem = cp.Problem(sub_objective, sub_constraints)
    
    # Solve
    print(f"Solving CVXPY relaxation for {component_size} nodes...")
    try:
        sub_problem.solve(solver=cp.ECOS, verbose=True)
        # sub_problem.solve(solver=cp.MOSEK, verbose=True)
        
        if sub_problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warning: Problem status is {sub_problem.status}")
            solution_y = np.zeros(component_size)
        else:
            solution_y = y.value
            print(f"CVXPY solution: objective = {sub_objective.value}")
    except Exception as e:
        print(f"Error solving CVXPY problem: {e}")
        solution_y = np.zeros(component_size)
    
    # Extract solution
    # Sort nodes by their values
    sorted_indices = np.argsort(-solution_y)
    
    # Try different thresholds
    best_sol = []
    best_size = 0
    
    print("Extracting discrete solution...")
    for k in range(1, component_size + 1):
        selected_indices = sorted_indices[:k]
        selected_nodes = [component_nodes[i] for i in selected_indices]
        
        if is_mostly_clique(G, selected_nodes) and len(selected_nodes) > best_size:
            best_sol = selected_nodes
            best_size = len(selected_nodes)
            print(f"Found solution of size {best_size} with density {compute_density(G, best_sol):.4f}")
    
    # Use this as a seed for greedy expansion
    print("Expanding solution greedily...")
    expanded = greedy_expansion(G, best_sol)
    
    return expanded

def sparse_frank_wolfe(G, max_iter=50, initial_size=1000):
    """Implement a sparse version of Frank-Wolfe optimized for the mostly-clique problem."""
    print("Setting up sparse Frank-Wolfe optimization...")
    
    # Create a subgraph with the highest degree nodes for initial tractability
    node_degrees = dict(G.degree())
    sorted_nodes = sorted(node_degrees.keys(), key=lambda n: node_degrees[n], reverse=True)
    subgraph_nodes = sorted_nodes[:initial_size]
    subgraph = G.subgraph(subgraph_nodes)
    
    # Get adjacency matrix
    A = nx.to_scipy_sparse_array(subgraph)
    n = len(subgraph_nodes)
    
    # Initial point: uniform weights
    x = np.ones(n) / n
    
    # Cache node neighbor information for efficient constraint computation
    node_neighbors = {}
    for i, node in enumerate(subgraph_nodes):
        node_neighbors[i] = set(G.neighbors(node)) & set(subgraph_nodes)
    
    # Optimize: Frank-Wolfe iterations
    print(f"Running Frank-Wolfe with {n} nodes for {max_iter} iterations...")
    for t in tqdm(range(max_iter)):
        # Compute gradient of the objective (sum of xi)
        obj_grad = np.ones(n)
        
        # Compute gradient of the constraint
        constraint_grad = np.zeros(n)
        
        # For each node, compute its contribution to the constraint
        for i in range(n):
            # Sum of edges involving node i
            edge_sum = sum(x[j] for j in node_neighbors[i])
            
            # Total potential connections
            total_connections = sum(x) - x[i]
            
            # Contribution to gradient: edges - 0.5 * all possible edges
            constraint_grad[i] = edge_sum - 0.5 * total_connections
        
        # Compute step size (decreasing with t)
        step_size = 2.0 / (t + 2)
        
        # Compute linear approximation minimizer
        s = np.zeros(n)
        sorted_indices = np.argsort(-constraint_grad)
        
        # Binary search to find largest k such that the top k nodes form a mostly-clique
        left, right = 1, n
        best_k = 1
        
        while left <= right:
            k = (left + right) // 2
            top_k_indices = sorted_indices[:k]
            top_k_nodes = [subgraph_nodes[i] for i in top_k_indices]
            
            if is_mostly_clique(subgraph, top_k_nodes):
                best_k = k
                left = k + 1
            else:
                right = k - 1
        
        # Set the top best_k nodes to 1
        s[sorted_indices[:best_k]] = 1.0
        
        # Update solution
        x = (1 - step_size) * x + step_size * s
    
    # Convert continuous solution to discrete
    print("Extracting discrete solution...")
    sorted_indices = np.argsort(-x)
    
    # Try different thresholds
    best_sol = []
    best_size = 0
    
    for k in range(1, n + 1):
        selected_indices = sorted_indices[:k]
        selected_nodes = [subgraph_nodes[i] for i in selected_indices]
        
        if is_mostly_clique(G, selected_nodes) and len(selected_nodes) > best_size:
            best_sol = selected_nodes
            best_size = len(selected_nodes)
            print(f"Found solution of size {best_size} with density {compute_density(G, best_sol):.4f}")
            
            # Limit search to avoid excessive computation
            if k > 100 and k > best_size + 20:
                break
    
    # Expand greedily
    print("Expanding solution greedily...")
    expanded = greedy_expansion(G, best_sol)
    
    return expanded

def find_large_mostly_clique(graph_file, output_file):
    """Main function to find a large mostly-clique using various methods."""
    start_time = time.time()
    
    # Load graph
    G = load_graph(graph_file)
    
    # First try CVXPY-based approach
    print("\n=== CVXPY Relaxation Approach ===")
    cvxpy_solution = relaxed_mostly_clique_cvxpy(G)
    cvxpy_size = len(cvxpy_solution)
    cvxpy_density = compute_density(G, cvxpy_solution)
    print(f"CVXPY solution: size = {cvxpy_size}, density = {cvxpy_density:.4f}")
    
    # Then try sparse Frank-Wolfe approach
    print("\n=== Sparse Frank-Wolfe Approach ===")
    fw_solution = sparse_frank_wolfe(G)
    fw_size = len(fw_solution)
    fw_density = compute_density(G, fw_solution)
    print(f"Frank-Wolfe solution: size = {fw_size}, density = {fw_density:.4f}")
    
    # Select the best solution
    if fw_size >= cvxpy_size:
        final_solution = fw_solution
        print(f"Using Frank-Wolfe solution: size = {fw_size}")
    else:
        final_solution = cvxpy_solution
        print(f"Using CVXPY solution: size = {cvxpy_size}")
    
    # Write solution to file
    with open(output_file, 'w') as f:
        f.write(' '.join(map(str, final_solution)))
    
    total_time = time.time() - start_time
    print(f"Solution written to {output_file}")
    print(f"Total runtime: {total_time:.2f} seconds")
    
    return final_solution

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python solver_mostly_clique.py <graph_file> <output_file>")
        sys.exit(1)
    
    graph_file = sys.argv[1]
    output_file = sys.argv[2]
    
    find_large_mostly_clique(graph_file, output_file)