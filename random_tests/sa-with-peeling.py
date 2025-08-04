import pandas as pd
import random
import time
import os
import gzip
import argparse # For command-line arguments

# --- Global Storage (remains the same) ---
adj_list = {}
edges_global_set = set()
all_node_ids = []
S_best = set()
k_best = 0
m_best = 0

# --- Helper Functions (load_graph_data, get_internal_degree, check_quasi_clique_condition, update_best_solution_global - remain largely the same) ---
# Minor addition: A robust edge counter for an arbitrary set of nodes
def count_edges_in_provided_subgraph(subgraph_nodes_set):
    """Counts edges within a given set of nodes, using the global_edges_set."""
    global edges_global_set # Ensure access to the global set
    if not subgraph_nodes_set or len(subgraph_nodes_set) < 2:
        return 0
    count = 0
    
    # Convert set to list for indexed iteration to form pairs
    nodes_list = list(subgraph_nodes_set)
    k_sub = len(nodes_list)

    for i in range(k_sub):
        for j in range(i + 1, k_sub):
            u, v = nodes_list[i], nodes_list[j]
            # Ensure canonical form (sorted tuple) for checking in edges_global_set
            if tuple(sorted((u,v))) in edges_global_set:
                count += 1
    return count

# --- load_graph_data, get_internal_degree, check_quasi_clique_condition, update_best_solution_global functions are assumed to be here, unchanged from your previous version ---
# Make sure they are defined before run_search_strategy

# COPYING THE PREVIOUSLY DEFINED HELPER FUNCTIONS FOR COMPLETENESS:
def load_graph_data(edge_file_path="edges.csv"):
    global adj_list, edges_global_set, all_node_ids
    print(f"Loading graph data from {edge_file_path}...")
    start_time = time.time()
    
    adj_list.clear()
    edges_global_set.clear()
    all_node_ids_list = [] # Use a temporary list for all_node_ids

    unique_nodes = set()
    try:
        with open(edge_file_path, 'rt') as f:
            for line_num, line in enumerate(f):
                # Basic header skip: if first line and no comma, or contains typical header words
                if line_num == 0 and (',' not in line or any(hdr_word in line.lower() for hdr_word in ['source', 'target', 'node', 'id'])):
                    try:
                        # Try to split and see if they are numeric, if not, likely a header
                        int(line.strip().split(',')[0]) 
                    except (ValueError, IndexError):
                        print(f"Skipping potential header line: {line.strip()}")
                        continue

                parts = line.strip().split(',')
                if len(parts) != 2:
                    # print(f"Warning: Malformed line {line_num+1}: {line.strip()} - Skipping.")
                    continue
                try:
                    u, v = int(parts[0]), int(parts[1])
                except ValueError:
                    # print(f"Warning: Non-integer node ID in line {line_num+1}: {line.strip()} - Skipping.")
                    continue
                
                unique_nodes.add(u)
                unique_nodes.add(v)
                
                adj_list.setdefault(u, set()).add(v)
                adj_list.setdefault(v, set()).add(u)
                
                edges_global_set.add(tuple(sorted((u, v))))
    except FileNotFoundError:
        print(f"Error: File {edge_file_path} not found.")
        return False
    except Exception as e:
        print(f"An error occurred during file reading: {e}")
        return False

    all_node_ids_list = sorted(list(unique_nodes)) # Assign to the correct global name
    globals()['all_node_ids'] = all_node_ids_list # Explicitly assign to global scope

    duration = time.time() - start_time
    print(f"Graph loaded: {len(all_node_ids)} nodes, {len(edges_global_set)} unique undirected edges. Time: {duration:.2f}s")
    return True

def get_internal_degree(node_id, subgraph_nodes_set):
    if node_id not in adj_list or not subgraph_nodes_set:
        return 0
    return len(adj_list.get(node_id, set()) & subgraph_nodes_set)

def check_quasi_clique_condition(k_val, m_val):
    if k_val < 2:
        return False
    return m_val * 8 > k_val * (k_val - 1)

def update_best_solution_global(current_S_set, current_m_val, current_k_val):
    global S_best, k_best, m_best
    is_updated = False
    if current_k_val > k_best:
        S_best = current_S_set.copy()
        k_best = current_k_val
        m_best = current_m_val
        is_updated = True
        density = (m_best / (k_best * (k_best - 1) / 2.0)) if k_best > 1 else 0
        print(f"NEW BEST: k={k_best}, m={m_best} (density: {density:.4f}, margin: {m_best - k_best*(k_best-1)/8.0:.2f})")
    elif current_k_val == k_best and current_k_val > 0: # If k is same, prefer higher margin (better quality)
        # Ensure k_val is not zero to prevent division by zero if k_val*(k_val-1) is zero
        current_margin = current_m_val - current_k_val*(current_k_val-1)/8.0
        best_margin = m_best - k_best*(k_best-1)/8.0 # m_best and k_best refer to the existing global best
        if current_margin > best_margin:
            S_best = current_S_set.copy()
            m_best = current_m_val
            is_updated = True
            density = (m_best / (k_best * (k_best - 1) / 2.0)) if k_best > 1 else 0
            print(f"QUALITY UPDATE: k={k_best}, m={m_best} (density: {density:.4f}, margin: {m_best - k_best*(k_best-1)/8.0:.2f})")
    return is_updated
# END OF COPIED HELPER FUNCTIONS


# --- New Function to Load Initial Solution ---
def load_initial_solution_nodes(solution_file_path):
    """Loads node IDs from a solution file (one ID per line)."""
    loaded_nodes = set()
    print(f"Attempting to load initial solution from: {solution_file_path}")
    try:
        # Determine if the file is gzipped
        is_gzipped = solution_file_path.endswith('.gz')
        open_func = gzip.open if is_gzipped else open
        
        with open_func(solution_file_path, 'rt') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'): # Skip empty lines or comments
                    continue
                # Check if the line could be a header (e.g. "node_id", "neuron_id")
                if line_num == 0 and not line.isdigit(): # crude header check
                    print(f"Skipping potential header in solution file: {line}")
                    continue
                try:
                    loaded_nodes.add(int(line))
                except ValueError:
                    print(f"Warning: Non-integer node ID in solution file line {line_num+1}: '{line}' - Skipping.")
                    continue
        print(f"Successfully loaded {len(loaded_nodes)} nodes from {solution_file_path}.")
        return loaded_nodes
    except FileNotFoundError:
        print(f"Error: Initial solution file '{solution_file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading initial solution: {e}")
        return None

# --- Modified Search Strategy ---
def run_search_strategy(num_main_iterations=10, max_local_search_steps=50, 
                          greedy_expansion_candidates_limit=200,
                          initial_solution_was_loaded=False): # Added flag
    global S_best, k_best, m_best 

    if not all_node_ids:
        print("Graph data not loaded. Call load_graph_data() first.")
        return

    for iteration in range(num_main_iterations):
        iter_time_start = time.time()
        print(f"\n--- Main Iteration {iteration + 1}/{num_main_iterations} ---")
        
        S_current = set()
        m_current = 0
        k_current = 0

        # A. Seed Initialization
        # If an initial solution was loaded and it's the first iteration, start with it or a perturbation.
        # Otherwise, use the general S_best perturbation or random seed.
        if iteration == 0 and initial_solution_was_loaded and S_best:
            S_current = S_best.copy() # Start directly with the loaded best solution
            m_current = m_best
            k_current = k_best
            print(f"Seed: Starting iteration 1 with loaded best solution (k={k_current}).")
        elif S_best and random.random() < 0.7 and k_best > 10 : # Increased chance to perturb S_best
            S_current = S_best.copy()
            nodes_to_remove_count = max(1, int(len(S_current) * random.uniform(0.05, 0.20))) # Perturb a bit more
            if len(S_current) > nodes_to_remove_count :
                # Ensure we don't try to sample more than available
                removable_nodes = list(S_current)
                nodes_removed_actual = random.sample(removable_nodes, min(nodes_to_remove_count, len(removable_nodes)))
                for r_node in nodes_removed_actual:
                    S_current.remove(r_node)
                # Recalculate m and k for the perturbed set
                k_current = len(S_current)
                m_current = count_edges_in_provided_subgraph(S_current) if k_current > 1 else 0
                print(f"Seed: Perturbed S_best (removed {len(nodes_removed_actual)} nodes). New start k={k_current}, m={m_current}")
            else: # S_best is too small to perturb meaningfully, start fresh
                seed_node = random.choice(all_node_ids)
                S_current = {seed_node}
                k_current = 1
                m_current = 0
                print(f"Seed: S_best too small to perturb, chose random node {seed_node}")

        else: # New random seed
            if not all_node_ids: # Should not happen if graph loaded
                 print("Error: all_node_ids is empty in seeding.")
                 continue
            seed_node = random.choice(all_node_ids)
            S_current = {seed_node}
            k_current = 1
            m_current = 0
            print(f"Seed: Random node {seed_node}")
        
        # Initial check of the current seed before expansion
        if check_quasi_clique_condition(k_current, m_current):
             update_best_solution_global(S_current, m_current, k_current)

        # B. Greedy Expansion Phase (remains the same logic as before)
        # print("  Starting Greedy Expansion Phase...")
        expanded_in_iteration = True
        while expanded_in_iteration:
            expanded_in_iteration = False
            best_candidate_to_add = None
            best_resulting_margin = -float('inf') 

            potential_additions_pool = set()
            for node_in_s in S_current: # S_current might be empty if perturbed S_best became empty
                if node_in_s in adj_list: # Check if node_in_s is in adj_list
                    for neighbor in adj_list.get(node_in_s, set()):
                        if neighbor not in S_current:
                            potential_additions_pool.add(neighbor)
            
            actual_candidates_to_check = list(potential_additions_pool)
            if len(potential_additions_pool) > greedy_expansion_candidates_limit :
                candidate_scores = []
                for cand_node in potential_additions_pool:
                    candidate_scores.append( (get_internal_degree(cand_node, S_current) , cand_node) )
                candidate_scores.sort(key=lambda x: x[0], reverse=True) 
                actual_candidates_to_check = [item[1] for item in candidate_scores[:greedy_expansion_candidates_limit]]

            if not actual_candidates_to_check:
                break

            k_new_expansion = k_current + 1
            for v_cand in actual_candidates_to_check:
                deg_S_v_cand = get_internal_degree(v_cand, S_current)
                m_new_expansion = m_current + deg_S_v_cand
                
                if check_quasi_clique_condition(k_new_expansion, m_new_expansion):
                    margin_new = m_new_expansion - k_new_expansion * (k_new_expansion - 1) / 8.0
                    if margin_new > best_resulting_margin:
                        best_resulting_margin = margin_new
                        best_candidate_to_add = v_cand
            
            if best_candidate_to_add:
                deg_S_added_node = get_internal_degree(best_candidate_to_add, S_current)
                S_current.add(best_candidate_to_add)
                m_current += deg_S_added_node
                k_current += 1
                expanded_in_iteration = True
                if check_quasi_clique_condition(k_current, m_current):
                     update_best_solution_global(S_current, m_current, k_current)
            else:
                break
        # print(f"  Greedy Expansion ended. k={k_current}, m={m_current}")

        # C. Local Search / Refinement Phase (Swaps - remains the same logic as before)
        # print("  Starting Local Search (Swaps)...")
        for local_step in range(max_local_search_steps):
            if k_current < 2 : break 

            nodes_in_S_list = list(S_current)
            if not nodes_in_S_list: break # S_current became empty
            u_remove = random.choice(nodes_in_S_list) 

            potential_v_add_pool = set()
            temp_S_minus_u = S_current - {u_remove}
            for node_in_s_minus_u in temp_S_minus_u:
                 if node_in_s_minus_u in adj_list: # Check key exists
                    for neighbor in adj_list.get(node_in_s_minus_u, set()):
                        if neighbor not in S_current : 
                            potential_v_add_pool.add(neighbor)
            
            if not potential_v_add_pool:
                # Consider all nodes not in S_current if no direct neighbors found for swap
                # This can be slow if all_node_ids is very large, sample instead
                all_others = set(all_node_ids) - S_current
                if not all_others : break
                potential_v_add_pool.update(random.sample(list(all_others), min(len(all_others), max(50, greedy_expansion_candidates_limit//2))))


            if not potential_v_add_pool: break
            v_add = random.choice(list(potential_v_add_pool)) 

            deg_u_in_S = get_internal_degree(u_remove, S_current) 
            m_after_remove = m_current - deg_u_in_S
            deg_v_to_S_minus_u = get_internal_degree(v_add, temp_S_minus_u)
            m_try_swap = m_after_remove + deg_v_to_S_minus_u
            k_try_swap = k_current 

            if check_quasi_clique_condition(k_try_swap, m_try_swap):
                # Using m_best and k_best for current_margin calculation is incorrect if we are improving S_current
                # It should be based on S_current's m and k
                current_qc_margin = m_current - k_current * (k_current - 1) / 8.0
                try_qc_margin = m_try_swap - k_try_swap * (k_try_swap - 1) / 8.0
                
                if try_qc_margin > current_qc_margin: 
                    S_current.remove(u_remove)
                    S_current.add(v_add)
                    m_current = m_try_swap
                    update_best_solution_global(S_current, m_current, k_current)
        
        iter_duration = time.time() - iter_time_start
        print(f"  Iteration {iteration+1} finished. Current S_current k={k_current}. Duration: {iter_duration:.2f}s. Overall best k={k_best}")

    print("\n--- Search Finished ---")
    if k_best > 0:
        density_final = (m_best / (k_best * (k_best - 1) / 2.0)) if k_best > 1 else 0
        print(f"Overall Best Solution: k = {k_best}, m = {m_best} (density: {density_final:.4f}, margin: {m_best - k_best*(k_best-1)/8.0:.2f})")
    else:
        print("No valid quasi-clique found or maintained.")
    return S_best


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Find large quasi-cliques in a graph.")
    parser.add_argument('--edges', type=str, default="edges.csv.gz", help="Path to the gzipped edge list file.")
    parser.add_argument('--initial_solution', type=str, default=None, help="Optional path to a file with node IDs for an initial solution.")
    parser.add_argument('--iterations', type=int, default=50, help="Number of main search iterations.")
    parser.add_argument('--local_search_steps', type=int, default=200, help="Max steps for local search (swaps) per iteration.")
    parser.add_argument('--expansion_candidates', type=int, default=500, help="Limit for candidates in greedy expansion.")
    
    args = parser.parse_args()

    active_edge_file = args.edges
    initial_solution_file = args.initial_solution
    num_iterations = args.iterations
    local_search_iter = args.local_search_steps
    expansion_limit = args.expansion_candidates
    initial_solution_loaded_successfully = False

    # Create a dummy edge file if specified one doesn't exist, for basic testing
    # (This part might be removed if you always expect the real file to exist)
    # This behavior is a bit unusual for a CLI tool, usually it would just error out.
    # For this challenge, let's assume the user will provide the correct edges.csv.gz
    if not os.path.exists(active_edge_file):
        print(f"ERROR: Edge file '{active_edge_file}' not found. Please provide a valid edge file.")
        exit(1) # Exit if the main data file is missing

    if load_graph_data(edge_file_path=active_edge_file):
        if initial_solution_file:
            loaded_nodes = load_initial_solution_nodes(initial_solution_file)
            if loaded_nodes and all_node_ids: # Ensure graph is loaded to verify nodes
                # Verify all loaded nodes actually exist in the graph
                valid_loaded_nodes = {node for node in loaded_nodes if node in adj_list or node in all_node_ids} # check against adj_list keys or all_node_ids
                if len(valid_loaded_nodes) != len(loaded_nodes):
                    print(f"Warning: {len(loaded_nodes) - len(valid_loaded_nodes)} nodes from the initial solution file were not found in the main graph and were discarded.")
                
                if valid_loaded_nodes:
                    k_loaded = len(valid_loaded_nodes)
                    m_loaded = count_edges_in_provided_subgraph(valid_loaded_nodes)
                    
                    print(f"Loaded initial solution: k={k_loaded}, m={m_loaded}.")
                    if check_quasi_clique_condition(k_loaded, m_loaded):
                        print("Initial solution is a valid quasi-clique.")
                        # Initialize S_best, k_best, m_best with this loaded solution
                        S_best = valid_loaded_nodes.copy()
                        k_best = k_loaded
                        m_best = m_loaded
                        initial_solution_loaded_successfully = True
                        # update_best_solution_global(valid_loaded_nodes, m_loaded, k_loaded) # This will print the "NEW BEST" message
                        print(f"Set as initial best: k={k_best}, m={m_best} (density: {(m_best / (k_best * (k_best-1)/2.0)) if k_best >1 else 0:.4f})")

                    else:
                        print("Warning: Loaded initial solution is NOT a valid quasi-clique according to the δ=0.25 criteria. It will not be used as a starting S_best.")
                else:
                    print("Warning: No valid nodes found in the initial solution file after checking against the graph. Proceeding with random seeds.")
            else:
                 print("Warning: Could not load initial solution or graph not loaded. Proceeding with random seeds.")


        found_solution_nodes = run_search_strategy(
            num_main_iterations=num_iterations, 
            max_local_search_steps=local_search_iter,
            greedy_expansion_candidates_limit=expansion_limit,
            initial_solution_was_loaded=initial_solution_loaded_successfully # Pass the flag
        ) 

        if k_best > 0:
            print(f"\nTo submit, save these {k_best} nodes to a file (first 10 shown if solution is large):")
            # sorted_solution_nodes = sorted(list(S_best))
            # for i, node_id in enumerate(sorted_solution_nodes):
            #     if i < 10: print(node_id)
            #     elif i == 10 and len(sorted_solution_nodes) > 10 : print("...")
            #     # No need to break, just print first 10 or all if less than 10
            
            solution_filename = f"quasi_clique_solution_k{k_best}_m{m_best}.csv"
            try:
                with open(solution_filename, 'w') as outfile:
                    # Ensure S_best (which is a set) is converted to a list and sorted for consistent output
                    for node_id in sorted(list(S_best)): 
                        outfile.write(f"{node_id}\n")
                print(f"Full solution saved to: {solution_filename}")
            except Exception as e:
                print(f"Error saving solution file: {e}")

        else:
            print("No solution found or improved during the search.")
    else:
        print("Could not load graph data. Aborting search.")