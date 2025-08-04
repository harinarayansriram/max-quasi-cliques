import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.sparse.linalg import eigs
import random
import time
import os
import gzip
import argparse

# --- Global Best Solution Tracking ---
S_BEST_DISCRETE = set()
K_BEST_DISCRETE = 0
M_BEST_DISCRETE = 0
DELTA_THRESHOLD = 0.25 # Given by the challenge

# --- Utility & Graph Loading ---
class GraphManager:
    def __init__(self, edge_file_path):
        self.adj_list = {}
        self.edges_global_set = set() # For quick edge existence checks: (u,v) with u<v
        self.all_node_ids_sorted = [] # Sorted list of all unique node IDs
        self.node_to_idx = {} # Map node ID to 0-N-1 index
        self.idx_to_node = {} # Map 0-N-1 index to node ID
        self.N = 0 # Number of unique nodes
        self.E = 0 # Number of unique edges (undirected)
        self._sparse_adj_matrix = None
        self._node_degrees = None

        self._load_graph(edge_file_path)

    def _load_graph(self, edge_file_path):
        print(f"Loading graph data from {edge_file_path}...")
        start_time = time.time()
        
        unique_nodes_set = set()
        temp_edges = set()

        try:
            with open(edge_file_path, 'rt') as f:
                for line_num, line in enumerate(f):
                    # Basic header skip
                    if line_num == 0:
                        try:
                            parts_test = line.strip().split(',')
                            int(parts_test[0])
                            int(parts_test[1])
                        except (ValueError, IndexError):
                            print(f"Skipping potential header line: {line.strip()}")
                            continue
                    
                    parts = line.strip().split(',')
                    if len(parts) != 2: continue
                    try:
                        u, v = int(parts[0]), int(parts[1])
                    except ValueError: continue
                    
                    unique_nodes_set.add(u)
                    unique_nodes_set.add(v)
                    
                    # Store edges temporarily to count unique ones later
                    temp_edges.add(tuple(sorted((u, v))))

            self.all_node_ids_sorted = sorted(list(unique_nodes_set))
            self.N = len(self.all_node_ids_sorted)
            for i, node_id in enumerate(self.all_node_ids_sorted):
                self.node_to_idx[node_id] = i
                self.idx_to_node[i] = node_id

            # Build adjacency list using mapped indices for potentially better cache locality later
            # However, for direct use with original IDs, mapping can be done on the fly or store original IDs
            # For now, let's keep adj_list with original IDs for clarity with other functions
            for u_orig, v_orig in temp_edges:
                self.adj_list.setdefault(u_orig, set()).add(v_orig)
                self.adj_list.setdefault(v_orig, set()).add(u_orig)
                self.edges_global_set.add(tuple(sorted((u_orig,v_orig)))) # Already sorted

            self.edges_global_set_indices = set() # For SMM m_c and discrete seeding
            for u_orig, v_orig in temp_edges: # temp_edges contains (u,v) with u<v
                # ... (adj_list and edges_global_set building) ...
                u_idx = self.node_to_idx[u_orig]
                v_idx = self.node_to_idx[v_orig]
                self.edges_global_set_indices.add(tuple(sorted((u_idx, v_idx))))

            self.E = len(self.edges_global_set)

        except FileNotFoundError:
            print(f"Error: File {edge_file_path} not found.")
            raise
        except Exception as e:
            print(f"An error occurred during graph file reading: {e}")
            raise

        duration = time.time() - start_time
        print(f"Graph loaded: {self.N} nodes, {self.E} unique undirected edges. Time: {duration:.2f}s")

    def get_sparse_adjacency_matrix(self):
        if self._sparse_adj_matrix is None:
            print("Building sparse adjacency matrix...")
            row_ind = []
            col_ind = []
            data = [] # Typically 1 for unweighted graph
            for u_orig, neighbors in self.adj_list.items():
                u_idx = self.node_to_idx[u_orig]
                for v_orig in neighbors:
                    v_idx = self.node_to_idx[v_orig]
                    # Add edges for both directions in sparse matrix if it's symmetric
                    row_ind.append(u_idx)
                    col_ind.append(v_idx)
                    data.append(1) 
            self._sparse_adj_matrix = sp.csr_matrix((data, (row_ind, col_ind)), shape=(self.N, self.N))
            print("Sparse adjacency matrix built.")
        return self._sparse_adj_matrix
    
    def get_node_degrees(self):
        if self._node_degrees is None:
            self._node_degrees = {node: len(neighbors) for node, neighbors in self.adj_list.items()}
        return self._node_degrees

class QuasiCliqueUtils:
    @staticmethod
    def check_condition(k, m, delta=DELTA_THRESHOLD):
        if k < 2: return False
        return m * (1/delta) > k * (k - 1) / 2.0 # Using 1/delta to avoid division by delta if it's 0
        # Original condition: m > delta * k * (k-1) / 2
        # For delta = 0.25, this is m > k(k-1)/8  or 8m > k(k-1)
    
    @staticmethod
    def count_internal_edges(nodes_set, graph_manager_instance):
        if not nodes_set or len(nodes_set) < 2: return 0
        count = 0
        # Iterate over pairs, check against global edge set for efficiency
        nodes_list = list(nodes_set)
        for i in range(len(nodes_list)):
            for j in range(i + 1, len(nodes_list)):
                u, v = nodes_list[i], nodes_list[j]
                if tuple(sorted((u,v))) in graph_manager_instance.edges_global_set:
                    count += 1
        return count

    @staticmethod
    def get_internal_degree(node_id, current_set, graph_manager_instance):
        if node_id not in graph_manager_instance.adj_list: return 0
        return len(graph_manager_instance.adj_list[node_id] & current_set)

    @staticmethod
    def update_global_best(current_S_set, current_m_val, current_k_val):
        global S_BEST_DISCRETE, K_BEST_DISCRETE, M_BEST_DISCRETE
        is_updated = False
        current_margin = current_m_val - DELTA_THRESHOLD * current_k_val * (current_k_val - 1) / 2.0 if current_k_val > 1 else -float('inf')
        
        if current_k_val > K_BEST_DISCRETE:
            S_BEST_DISCRETE = current_S_set.copy()
            K_BEST_DISCRETE = current_k_val
            M_BEST_DISCRETE = current_m_val
            is_updated = True
            density = (M_BEST_DISCRETE / (K_BEST_DISCRETE * (K_BEST_DISCRETE - 1) / 2.0)) if K_BEST_DISCRETE > 1 else 0
            print(f"NEW BEST: k={K_BEST_DISCRETE}, m={M_BEST_DISCRETE} (density: {density:.4f}, margin: {current_margin:.2f})")
        elif current_k_val == K_BEST_DISCRETE and current_k_val > 0:
            best_margin = M_BEST_DISCRETE - DELTA_THRESHOLD * K_BEST_DISCRETE * (K_BEST_DISCRETE - 1) / 2.0 if K_BEST_DISCRETE > 1 else -float('inf')
            if current_margin > best_margin:
                S_BEST_DISCRETE = current_S_set.copy()
                M_BEST_DISCRETE = current_m_val
                is_updated = True
                density = (M_BEST_DISCRETE / (K_BEST_DISCRETE * (K_BEST_DISCRETE - 1) / 2.0)) if K_BEST_DISCRETE > 1 else 0
                print(f"QUALITY UPDATE: k={K_BEST_DISCRETE}, m={M_BEST_DISCRETE} (density: {density:.4f}, margin: {current_margin:.2f})")
        return is_updated

    @staticmethod
    def load_initial_solution_nodes(solution_file_path, graph_manager_instance):
        loaded_nodes = set()
        print(f"Attempting to load initial solution from: {solution_file_path}")
        try:
            is_gzipped = solution_file_path.endswith('.gz')
            open_func = gzip.open if is_gzipped else open
            
            with open_func(solution_file_path, 'rt') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    if line_num == 0 and not line.isdigit():
                        print(f"Skipping potential header in solution file: {line}")
                        continue
                    try:
                        node_id = int(line)
                        if node_id in graph_manager_instance.node_to_idx: # Check if node exists in graph
                             loaded_nodes.add(node_id)
                        else:
                            print(f"Warning: Node ID {node_id} from solution file not found in graph. Skipping.")
                    except ValueError:
                        print(f"Warning: Non-integer node ID in solution file line {line_num+1}: '{line}' - Skipping.")
            
            if loaded_nodes:
                k_loaded = len(loaded_nodes)
                m_loaded = QuasiCliqueUtils.count_internal_edges(loaded_nodes, graph_manager_instance)
                print(f"Successfully loaded {k_loaded} valid nodes from {solution_file_path} with {m_loaded} internal edges.")
                if QuasiCliqueUtils.check_condition(k_loaded, m_loaded):
                    print("Loaded solution is a valid quasi-clique. Setting as initial best.")
                    QuasiCliqueUtils.update_global_best(loaded_nodes, m_loaded, k_loaded)
                    return True # Indicates successful load and set
                else:
                    print("Warning: Loaded solution is NOT a valid quasi-clique. It will not be set as initial best.")
            else:
                print("No valid nodes loaded from the solution file.")

        except FileNotFoundError:
            print(f"Error: Initial solution file '{solution_file_path}' not found.")
        except Exception as e:
            print(f"An error occurred while loading initial solution: {e}")
        return False


# --- Smooth Margin Maximization (SMM) ---
class SmoothMarginOptimizer:
    def __init__(self, graph_manager, delta=DELTA_THRESHOLD):
        self.graph_manager = graph_manager
        self.delta = delta
        self.node_list_for_indexing = self.graph_manager.all_node_ids_sorted # for m_c calculation
        self.adj_list_indexed = self._get_adj_list_indexed() # Precompute for speed

    def _get_adj_list_indexed(self):
        # Create an adjacency list where nodes are represented by their 0 to N-1 indices
        adj_indexed = [[] for _ in range(self.graph_manager.N)]
        for u_orig, neighbors in self.graph_manager.adj_list.items():
            u_idx = self.graph_manager.node_to_idx[u_orig]
            for v_orig in neighbors:
                v_idx = self.graph_manager.node_to_idx[v_orig]
                adj_indexed[u_idx].append(v_idx)
        return adj_indexed

    def _objective_and_gradient(self, x_vector_flat):
        # x_vector_flat is a 1D numpy array of node weights (0 to 1)
        
        k_c = np.sum(x_vector_flat)
        if k_c < 1e-6: # Avoid division by zero or tiny k_c issues
            return 1e9, np.zeros_like(x_vector_flat) # Return high cost if no nodes selected

        m_c = 0.0
        # Efficient m_c: sum_{u<v, (u,v) in E} x_u * x_v
        # Iterate through edges once
        for u_orig, v_orig in self.graph_manager.edges_global_set:
            idx_u = self.graph_manager.node_to_idx[u_orig]
            idx_v = self.graph_manager.node_to_idx[v_orig]
            m_c += x_vector_flat[idx_u] * x_vector_flat[idx_v]

        # Objective: Maximize J(x) = m_c(x) - δ * k_c(x) * (k_c(x) - 1) / 2
        # For scipy.optimize.minimize, we minimize -J(x)
        objective_val = -(m_c - self.delta * k_c * (k_c - 1.0) / 2.0)

        # Gradient of -J(x) w.r.t x_j:
        # d(-J)/dx_j = - [ (sum_{v adj to j} x_v) - δ * (k_c - 1/2) ]
        grad = np.zeros_like(x_vector_flat)
        term_delta_kc = self.delta * (k_c - 0.5)
        
        for j_idx in range(self.graph_manager.N):
            sum_adj_x_v = 0.0
            for neighbor_idx in self.adj_list_indexed[j_idx]: # Use precomputed indexed adj list
                sum_adj_x_v += x_vector_flat[neighbor_idx]
            grad[j_idx] = -(sum_adj_x_v - term_delta_kc)
            
        # Add a small L2 regularization to push x towards 0 or 1 (optional, helps stability)
        # This encourages x_i * (1-x_i) to be small.
        lambda_reg = 0.5 #TODO: increase lambda
        objective_val += lambda_reg * np.sum(x_vector_flat * (1 - x_vector_flat))
        grad += lambda_reg * (1 - 2 * x_vector_flat)
            
        return objective_val, grad

    def optimize(self, initial_x_vector, max_smm_iter=100):
        print("Starting Smooth Margin Maximization (SMM)...")
        bounds = [(0.0, 1.0) for _ in range(self.graph_manager.N)]
        
        smm_start_time = time.time()
        result = minimize(self._objective_and_gradient, 
                          initial_x_vector, 
                          method='L-BFGS-B', 
                          jac=True, # We provide the gradient
                          bounds=bounds,
                          options={'maxiter': max_smm_iter, 'disp': True, 'ftol': 1e-7, 'gtol': 1e-5})
        smm_duration = time.time() - smm_start_time
        
        if result.success:
            print(f"SMM optimization finished successfully in {smm_duration:.2f}s. Final objective: {-result.fun:.4f} ({result.nit} iterations)")
        else:
            print(f"SMM optimization finished (not necessarily converged) in {smm_duration:.2f}s. Status: {result.message} ({result.nit} iterations)")
        
        return result.x # The optimized x_vector


# --- Spectral Analysis (Simplified for now) ---
class SpectralAnalyzer:
    def __init__(self, graph_manager):
        self.graph_manager = graph_manager

    def get_leading_eigenvectors(self, num_eigenvectors=5):
        print(f"Computing {num_eigenvectors} leading eigenvectors of the adjacency matrix...")
        adj_matrix_sparse = self.graph_manager.get_sparse_adjacency_matrix()
        
        # Using 'LM' for largest magnitude. For symmetric matrices, these are largest real eigenvalues.
        # 'which' can be 'LA' for largest algebraic if matrix is known to be positive semi-definite.
        try:
            eigenvalues, eigenvectors = eigs(adj_matrix_sparse, k=num_eigenvectors, which='LM', tol=1e-4, maxiter=self.graph_manager.N*2)
            # eigenvectors are columns, v[:,i] is the i-th eigenvector
            # We want real parts if complex (though for symmetric A, they should be real)
            return np.real(eigenvectors) 
        except Exception as e:
            print(f"Eigenvector computation failed: {e}")
            return None

# --- Discrete Search & Projection ---
class DiscreteOptimizer:
    def __init__(self, graph_manager, utils_class):
        self.graph_manager = graph_manager
        self.utils = utils_class

    def project_and_refine_solution(self, current_best_set, guidance_x_star,
                                     max_refine_steps=1000, k_target_min_factor=0.8, k_target_max_factor=1.2):
        print("Projecting continuous solution and refining discretely...")
        
        # Sort nodes by their x_star values (guidance scores)
        sorted_node_indices = np.argsort(-guidance_x_star) # Descending sort of scores
        
        candidate_set = set()
        current_m = 0
        current_k = 0

        # --- MODIFIED GREEDY CONSTRUCTION ---
        print("  Greedy/Seeded construction based on x_star scores...")
        
        # Determine a reasonable initial chunk size
        # If K_BEST_DISCRETE is substantial, use it as a hint. Otherwise, a fixed moderate size.
        initial_chunk_size = K_BEST_DISCRETE if K_BEST_DISCRETE > 10 else max(2, int(self.graph_manager.N * 0.001)) # e.g., 0.1% of N or at least 2
        initial_chunk_size = min(initial_chunk_size, self.graph_manager.N) # Cap at N

        # Take the top 'initial_chunk_size' nodes according to x_star
        potential_initial_nodes = {self.graph_manager.idx_to_node[idx] for idx in sorted_node_indices[:initial_chunk_size]}
        
        # If current_best_set is non-empty and good, consider starting refinement from it directly,
        # or a hybrid. For now, let's try to build from x_star first, then refine.
        
        # Option 1: Build from scratch using top x_star nodes with a more robust greedy add
        # This is more robust than adding one by one to an empty set.
        # Iteratively add nodes from `potential_initial_nodes` or even `sorted_node_indices`
        # if they improve or maintain a valid quasi-clique with a positive margin.

        # Let's try a more robust greedy add from sorted_node_indices:
        temp_candidate_set = set()
        temp_m, temp_k = 0, 0
        
        # First, find the best single edge based on x_star to ensure k>=2
        best_seed_edge_score = -1
        seed_edge_nodes = None
        if self.graph_manager.N >= 2:
            for u_idx, v_idx in np.array(list(self.graph_manager.edges_global_set_indices)): # Assuming you make this
                score = guidance_x_star[u_idx] + guidance_x_star[v_idx]
                if score > best_seed_edge_score:
                    best_seed_edge_score = score
                    seed_edge_nodes = {self.graph_manager.idx_to_node[u_idx], self.graph_manager.idx_to_node[v_idx]}
        
        if seed_edge_nodes:
            temp_candidate_set = seed_edge_nodes
            temp_k = 2
            temp_m = 1 # since it's an edge
            print(f"    Seeded greedy construction with edge: {temp_candidate_set}, k={temp_k}, m={temp_m}. QC: {self.utils.check_condition(temp_k, temp_m)}")
            if self.utils.check_condition(temp_k, temp_m):
                 self.utils.update_global_best(temp_candidate_set, temp_m, temp_k)
        else: # Fallback if no edges or N < 2
            print("    Could not seed with an edge for greedy construction.")


        # Now, try to expand this seed or build if seed failed
        for node_idx_to_add in sorted_node_indices:
            node_id_to_add = self.graph_manager.idx_to_node[node_idx_to_add]
            if node_id_to_add in temp_candidate_set: # Already added or part of seed
                continue

            deg_to_cand = self.utils.get_internal_degree(node_id_to_add, temp_candidate_set, self.graph_manager)
            
            k_new = temp_k + 1
            m_new = temp_m + deg_to_cand
            
            # We want to add if it's a valid QC *and* ideally doesn't shrink the margin too much, or increases k
            if self.utils.check_condition(k_new, m_new):
                # Optional: check if margin improves or k increases while valid
                # current_margin = temp_m - DELTA_THRESHOLD * temp_k * (temp_k-1)/2.0 if temp_k > 1 else -float('inf')
                # new_margin = m_new - DELTA_THRESHOLD * k_new * (k_new-1)/2.0
                # if new_margin >= current_margin or k_new > temp_k: # Or some other heuristic
                temp_candidate_set.add(node_id_to_add)
                temp_k = k_new
                temp_m = m_new
                self.utils.update_global_best(temp_candidate_set, temp_m, temp_k)
            # else: # If adding this node makes it invalid, try adding a different one.
                  # # More advanced: Could try adding it anyway if x_star is very high and then repair.
                #   pass
        
        candidate_set = temp_candidate_set # This is the result of the greedy construction
        current_k = temp_k
        current_m = temp_m
        # --- END OF MODIFIED GREEDY CONSTRUCTION ---
        
        print(f"  Greedy construction yielded k={current_k}, m={current_m}.")
        
        # Ensure active_set for local search is initialized properly
        if current_k > 0 and self.utils.check_condition(current_k, current_m):
            active_set = candidate_set.copy()
            active_k = current_k
            active_m = current_m
            print(f"  Proceeding to local search with set from greedy construction (k={active_k}).")
        elif K_BEST_DISCRETE > 0: # Fallback to global best if greedy failed
            active_set = S_BEST_DISCRETE.copy()
            active_k = K_BEST_DISCRETE
            active_m = M_BEST_DISCRETE
            print(f"  Greedy construction failed or invalid. Proceeding to local search with global best (k={active_k}).")
        else: # No valid set anywhere
            print("  No valid set from greedy or global best. Local search might be limited.")
            active_set = set() # Start local search with empty or a new small seed
            active_k = 0
            active_m = 0

        if active_k == 0 and self.graph_manager.N > 0: # If active_set is still empty
            print("  Active set for local search is empty. Attempting to re-seed locally.")
            # Try to seed with a random edge or a high x_star node
            if self.graph_manager.E > 0 :
                # Pick a random edge to start
                # random_edge_orig = random.choice(list(self.graph_manager.edges_global_set))
                # Or pick an edge based on x_star scores, similar to above
                best_seed_edge_score_local = -1
                seed_edge_nodes_local = None
                if self.graph_manager.N >= 2:
                    for u_idx, v_idx in np.array(list(self.graph_manager.edges_global_set_indices)):
                        score = guidance_x_star[u_idx] + guidance_x_star[v_idx]
                        if score > best_seed_edge_score_local:
                            best_seed_edge_score_local = score
                            seed_edge_nodes_local = {self.graph_manager.idx_to_node[u_idx], self.graph_manager.idx_to_node[v_idx]}
                
                if seed_edge_nodes_local:
                    active_set = seed_edge_nodes_local
                    active_k = 2
                    active_m = 1 # it's an edge
                    print(f"    Re-seeded local search with edge {active_set} based on x_star.")
                    if self.utils.check_condition(active_k, active_m):
                        self.utils.update_global_best(active_set, active_m, active_k)
                # else fallback to a single high x_star node if all else fails
                elif self.graph_manager.N > 0:
                    best_single_node_idx = np.argmax(guidance_x_star)
                    active_set = {self.graph_manager.idx_to_node[best_single_node_idx]}
                    active_k = 1
                    active_m = 0
                    print(f"    Re-seeded local search with single highest x_star node.")

        print(f"  Starting local refinement from k={active_k}, m={active_m} for {max_refine_steps} steps.")
        for step in range(max_refine_steps):
            if active_k == 0: break
            
            # Try a swap, or add, or remove, biased by x_star
            # Simplified: try one random swap or one random add/remove
            action_type = random.choice(['swap', 'add', 'remove'])
            
            new_set = active_set.copy()
            new_k, new_m = active_k, active_m

            if action_type == 'swap' and active_k > 0:
                # Remove node with low x_star from active_set or low internal degree
                # For simplicity: random for now, but better to choose intelligently
                # node_to_remove_id = min(active_set, key=lambda node_id: guidance_x_star[self.graph_manager.node_to_idx[node_id]])
                if not active_set: continue
                node_to_remove_id = random.choice(list(active_set))
                
                # Add node with high x_star not in active_set
                potential_add_candidates = {idx: guidance_x_star[idx] for idx in range(self.graph_manager.N) 
                                             if self.graph_manager.idx_to_node[idx] not in active_set}
                if not potential_add_candidates: continue
                # Weighted random choice based on x_star values, or just pick best
                node_to_add_idx = max(potential_add_candidates, key=potential_add_candidates.get)
                node_to_add_id = self.graph_manager.idx_to_node[node_to_add_idx]

                if node_to_add_id == node_to_remove_id: continue

                # Calculate effect of swap
                deg_remove_in_active = self.utils.get_internal_degree(node_to_remove_id, active_set - {node_to_remove_id}, self.graph_manager)
                temp_set_after_remove = active_set - {node_to_remove_id}
                deg_add_to_temp = self.utils.get_internal_degree(node_to_add_id, temp_set_after_remove, self.graph_manager)
                
                new_k = active_k # k is same for swap
                new_m = active_m - deg_remove_in_active + deg_add_to_temp
                
                if self.utils.check_condition(new_k, new_m):
                    current_margin = active_m - DELTA_THRESHOLD * active_k*(active_k-1)/2.0
                    new_margin = new_m - DELTA_THRESHOLD * new_k*(new_k-1)/2.0
                    if new_margin > current_margin : # Accept if improves margin
                        active_set.remove(node_to_remove_id)
                        active_set.add(node_to_add_id)
                        active_m = new_m
                        # active_k is unchanged
                        self.utils.update_global_best(active_set, active_m, active_k)


            elif action_type == 'add':
                potential_add_candidates = {idx: guidance_x_star[idx] for idx in range(self.graph_manager.N) 
                                             if self.graph_manager.idx_to_node[idx] not in active_set}
                if not potential_add_candidates: continue
                node_to_add_idx = max(potential_add_candidates, key=potential_add_candidates.get)
                node_to_add_id = self.graph_manager.idx_to_node[node_to_add_idx]

                deg_add = self.utils.get_internal_degree(node_to_add_id, active_set, self.graph_manager)
                k_try = active_k + 1
                m_try = active_m + deg_add
                if self.utils.check_condition(k_try, m_try):
                    active_set.add(node_to_add_id)
                    active_k = k_try
                    active_m = m_try
                    self.utils.update_global_best(active_set, active_m, active_k)

            elif action_type == 'remove' and active_k > 1:
                if not active_set : continue
                # Remove node with low x_star from active_set
                node_to_remove_idx = min(active_set, key=lambda node_id: guidance_x_star[self.graph_manager.node_to_idx[node_id]])
                node_to_remove_id = node_to_remove_idx
                
                deg_remove = self.utils.get_internal_degree(node_to_remove_id, active_set - {node_to_remove_id}, self.graph_manager)
                k_try = active_k - 1
                m_try = active_m - deg_remove
                if self.utils.check_condition(k_try, m_try) or k_try==0: # Allow removal if it keeps it valid or empties
                    active_set.remove(node_to_remove_id)
                    active_k = k_try
                    active_m = m_try
                    if k_try > 0 : self.utils.update_global_best(active_set, active_m, active_k)

            if step % (max_refine_steps // 10 + 1) == 0:
                print(f"    Refinement step {step}, current active k={active_k}, m={active_m}. Global best k={K_BEST_DISCRETE}")
        
        print(f"  Local refinement finished. Final active k={active_k}, m={active_m}.")
        return K_BEST_DISCRETE # Return the current global best k found during this process


# --- Main Iterative Loop ---
class IterativeSolver:
    def __init__(self, graph_manager, initial_solution_nodes=None, num_outer_loops=10, smm_iters=50, discrete_refine_steps=500):
        global S_BEST_DISCRETE, K_BEST_DISCRETE, M_BEST_DISCRETE

        self.graph_manager = graph_manager
        self.utils = QuasiCliqueUtils() # Static methods, no instance needed if designed so
        self.smm_optimizer = SmoothMarginOptimizer(graph_manager)
        # self.spectral_analyzer = SpectralAnalyzer(graph_manager) # Optional
        self.discrete_optimizer = DiscreteOptimizer(graph_manager, self.utils)
        
        self.num_outer_loops = num_outer_loops
        self.smm_iters = smm_iters
        self.discrete_refine_steps = discrete_refine_steps

        if initial_solution_nodes:
            # Global best is updated by load_initial_solution_nodes if valid
            print(f"Using provided initial solution for S_BEST_DISCRETE (k={K_BEST_DISCRETE})")
        else: # Start with a random high-degree node if no solution provided
            if not S_BEST_DISCRETE and self.graph_manager.N > 0:
                print("No initial solution provided or loaded. Seeding with a high-degree node.")
                degrees = self.graph_manager.get_node_degrees()
                if degrees:
                    # Sort nodes by degree, pick one from top (e.g. top 10 random)
                    sorted_nodes_by_degree = sorted(degrees.items(), key=lambda item: item[1], reverse=True)
                    top_nodes = [n for n,d in sorted_nodes_by_degree[:max(10, self.graph_manager.N // 1000)]] # Top 0.1% or 10
                    if top_nodes:
                        seed_node = random.choice(top_nodes)
                        S_BEST_DISCRETE.add(seed_node)
                        K_BEST_DISCRETE = 1
                        M_BEST_DISCRETE = 0
                        print(f"Seeded with node {seed_node}, k=1.")
                    else:
                        print("Could not seed with high-degree node (no nodes or degrees).")
                else:
                     print("Could not seed with high-degree node (degrees not computed).")


    def run(self):
        global S_BEST_DISCRETE, K_BEST_DISCRETE # To ensure we're using the global ones

        if self.graph_manager.N == 0:
            print("Graph is empty. Cannot run solver.")
            return

        for loop_iter in range(self.num_outer_loops):
            print(f"\n--- Outer Loop Iteration {loop_iter + 1}/{self.num_outer_loops} ---")
            print(f"Starting with best discrete solution: k={K_BEST_DISCRETE}, m={M_BEST_DISCRETE}")

            # 1. Initialize SMM from current S_BEST_DISCRETE
            # initial_x = np.full(self.graph_manager.N, 0.05) # Background value
            # if S_BEST_DISCRETE:
            #     for node_id in S_BEST_DISCRETE:
            #         idx = self.graph_manager.node_to_idx[node_id]
            #         initial_x[idx] = 0.95 # Strong signal for current solution members
            # else: # If S_BEST_DISCRETE is empty, maybe start with a more uniform or random x
            #      if self.graph_manager.N > 0: initial_x = np.random.uniform(0.2, 0.8, self.graph_manager.N)
            # 1. Initialize SMM from current S_BEST_DISCRETE
            initial_x = np.full(self.graph_manager.N, 0.1) # Smaller background
            if K_BEST_DISCRETE > 1 : # Only use S_BEST_DISCRETE if it's somewhat substantial
                for node_id in S_BEST_DISCRETE:
                    idx = self.graph_manager.node_to_idx[node_id]
                    initial_x[idx] = 0.9 # Stronger signal
            elif self.graph_manager.N > 0: # If S_BEST_DISCRETE is empty or just 1 node
                 # Initialize with a few top-degree nodes or random high x_star from previous if available
                 # For now, a slightly higher uniform random, or based on global degrees
                print("  SMM initial_x: S_BEST_DISCRETE is small/empty. Using a more uniform random init for x.")
                degrees = np.array([self.graph_manager.adj_list.get(self.graph_manager.idx_to_node[i], set()).__len__() for i in range(self.graph_manager.N)])
                if np.max(degrees) > 0:
                    initial_x = 0.1 + 0.8 * (degrees / np.max(degrees)) # Scale degrees to 0.1-0.9
                else:
                    initial_x = np.random.uniform(0.3, 0.7, self.graph_manager.N)

            # 2. Run SMM
            x_star_continuous = self.smm_optimizer.optimize(initial_x, max_smm_iter=self.smm_iters)

            # (Optional: Spectral guidance could modify x_star_continuous or initial_x here)
            # e.g. spectral_vectors = self.spectral_analyzer.get_leading_eigenvectors()
            # if spectral_vectors is not None: ... blend with x_star or use for biasing ...

            # 3. Project and Refine
            # Pass current S_BEST_DISCRETE so refinement can try to improve it or start from it
            self.discrete_optimizer.project_and_refine_solution(
                S_BEST_DISCRETE.copy() if S_BEST_DISCRETE else set(), 
                x_star_continuous,
                max_refine_steps=self.discrete_refine_steps
            )
            
            print(f"End of Outer Loop {loop_iter + 1}. Current Global Best: k={K_BEST_DISCRETE}, m={M_BEST_DISCRETE}")
            
            # Save intermediate best solution
            if K_BEST_DISCRETE > 0:
                solution_filename = f"quasi_clique_solution_iter{loop_iter+1}_k{K_BEST_DISCRETE}.csv"
                try:
                    with open(solution_filename, 'w') as outfile:
                        for node_id in sorted(list(S_BEST_DISCRETE)):
                            outfile.write(f"{node_id}\n")
                    print(f"Intermediate solution saved to: {solution_filename}")
                except Exception as e:
                    print(f"Error saving intermediate solution: {e}")


        print("\n--- Iterative Solver Finished ---")
        print(f"Final Best Solution: k={K_BEST_DISCRETE}, m={M_BEST_DISCRETE}")
        # Final save
        if K_BEST_DISCRETE > 0:
            final_solution_filename = f"quasi_clique_FINAL_k{K_BEST_DISCRETE}_m{M_BEST_DISCRETE}.csv"
            try:
                with open(final_solution_filename, 'w') as outfile:
                     for node_id in sorted(list(S_BEST_DISCRETE)):
                        outfile.write(f"{node_id}\n")
                print(f"Final solution saved to: {final_solution_filename}")
            except Exception as e:
                print(f"Error saving final solution: {e}")

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Find large quasi-cliques using iterative continuous-discrete optimization.")
    parser.add_argument('--edges', type=str, required=True, help="Path to the gzipped edge list file (e.g., edges.csv.gz).")
    parser.add_argument('--initial_solution', type=str, default=None, help="Optional path to a file with node IDs for an initial solution (k=369 benchmark).")
    parser.add_argument('--loops', type=int, default=10, help="Number of outer iterative refinement loops.")
    parser.add_argument('--smm_iter', type=int, default=75, help="Max iterations for SMM L-BFGS-B optimizer per loop.")
    parser.add_argument('--discrete_steps', type=int, default=2000, help="Max steps for discrete refinement per loop.")
    
    args = parser.parse_args()

    if not os.path.exists(args.edges):
        print(f"ERROR: Edge file '{args.edges}' not found. Please provide a valid edge file.")
        exit(1)

    try:
        graph_data_manager = GraphManager(args.edges)
        
        initial_solution_loaded = False
        if args.initial_solution:
            if os.path.exists(args.initial_solution):
                initial_solution_loaded = QuasiCliqueUtils.load_initial_solution_nodes(args.initial_solution, graph_data_manager)
            else:
                print(f"Warning: Initial solution file '{args.initial_solution}' not found. Starting without it.")

        solver = IterativeSolver(
            graph_data_manager, 
            initial_solution_nodes=S_BEST_DISCRETE if initial_solution_loaded else None, # Pass the set if loaded
            num_outer_loops=args.loops,
            smm_iters=args.smm_iter,
            discrete_refine_steps=args.discrete_steps
        )
        solver.run()

    except Exception as e:
        print(f"An critical error occurred in the main execution: {e}")
        import traceback
        traceback.print_exc()