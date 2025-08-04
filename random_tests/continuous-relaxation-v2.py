import jax
import jax.numpy as jnp
import optax

edges_file = "edges.csv"
to_compressed_idx: dict[int, int] = {}
from_compressed_idx: list[int] = []
adj: dict[int,set[int]] = {}
edge_idx_pairs = []

def load_edges(filename: str):
    global to_compressed_idx, from_compressed_idx, adj, edge_idx_pairs
    count = 0
    edges_read = 0
    with open(filename, encoding='utf-8') as f:
        while line := f.readline():
            edges_read += 1
            n1, n2 = map(int, line.strip().split(','))
            for n in (n1, n2):
                if n not in to_compressed_idx:
                    to_compressed_idx[n] = count
                    from_compressed_idx.append(n)
                    adj[count] = set()
                    count += 1

            adj[to_compressed_idx[n1]].add(to_compressed_idx[n2])
            adj[to_compressed_idx[n2]].add(to_compressed_idx[n1])

            edge_idx_pairs.append((to_compressed_idx[n1], to_compressed_idx[n2]))

    edge_idx_pairs = jnp.array(edge_idx_pairs)

    print(f'Read in graph: {edges_read} edges and {count} nodes')

def inverse_sigmoid(v):
    return jnp.log(v) - jnp.log(1-v)

def load_solution(filename: str):
    with open(filename, encoding='utf-8') as f:
        probabilities = jnp.array([float(x) for x in f.read().strip().split(',')])
        return inverse_sigmoid(probabilities)


def objective(u_i, delta, k_weight, sparse_weight, num_nodes):
    # using u_i's for unconstrained optimization, the sigmoid forces it naturally into 0-1 range for the actual v
    v = jax.nn.sigmoid(u_i)

    k_continous = jnp.sum(v)

    def effective_edges(edge_idxes):
        i, j = edge_idxes
        return v[i] * v[j]

    edges_continuous = jnp.sum(jax.lax.map(effective_edges, edge_idx_pairs))

    # maximize k_continous subject to edges_continous > delta * (k_continous choose 2)
    # so when condition is met, objective should be approx equal to some linear function of k_continuous; otherwise, objective should decrease rapidly as condition gets further from being met
    margin = edges_continuous - delta * k_continous * (k_continous - 1.0) / 2.0
    size = k_weight * k_continous
    sparsity = sparse_weight * jnp.sum(v*(1.0-v)) / num_nodes

    return -(margin + size - sparsity)


load_edges(edges_file)

solution_file = 'resulting_v_2.csv'


num_nodes = len(to_compressed_idx)
num_steps = 200
w_k = 0.5
w_s = 50

initial_u_params = jnp.array([])
if not solution_file:
    initial_u_params = jax.random.uniform(key=jax.random.key(42), minval=-2.0, maxval=2.0, shape=(num_nodes,))
else:
    initial_u_params = load_solution(solution_file)

optimizer = optax.adam(learning_rate=1e-1)
opt_state = optimizer.init(initial_u_params)
params = initial_u_params

obj_val_and_grad_fn = jax.value_and_grad(objective)

def save(v):
    with open("resulting_v_3.csv", "w") as f:
        f.write(','.join(map(str, v.tolist())))

# for step in range(num_steps):
#     loss_value, grads = obj_val_and_grad_fn(params, 0.25, w_k, w_s, num_nodes)
#     updates, opt_state = optimizer.update(grads, opt_state, params)
#     params = optax.apply_updates(params, updates)
#     if step % 1 == 0:
#         print(f"Step {step}, Loss: {loss_value}")

#     if step % 5 == 0:
#         print(f'Saving at step {step}')
#         save(jax.nn.sigmoid(params))

# final_v = jax.nn.sigmoid(params) # Get the v_i values
# save(final_v)
# print(final_v.tolist()[:100])
# print(jnp.histogram(final_v, bins=10))



from scipy.optimize import linprog
import numpy as np
import scipy.sparse as sp # For constructing sparse matrices for the LP
import math # For floor

# Assume these are available from your JAX script's context after optimization:
# final_v_numpy: np.array, shape (num_nodes,) - the fuzzy scores
# num_nodes: int - total number of nodes (len(to_compressed_idx))
# edge_idx_pairs_numpy: np.array or list of tuples, shape (num_edges, 2) 
#                       - containing (u_idx, v_idx) for each edge, using compressed 0-N-1 indices.
# You have: edge_idx_pairs (JAX array) -> convert to NumPy for this SciPy function
#           adj (dict[int, set[int]]) - uses compressed indices

def lp_project_custom_vars(final_v_numpy, k_target, 
                           num_nodes, edge_idx_pairs_numpy_list, 
                           delta_thresh=0.25, time_limit_lp=300):
    """
    Projects a fuzzy solution to a discrete one using LP for a fixed k_target.

    Args:
        final_v_numpy (np.array): Fuzzy membership scores (0 to 1) for each node.
        k_target (int): The target size for the discrete quasi-clique.
        num_nodes (int): Total number of nodes.
        edge_idx_pairs_numpy_list (list): List of tuples (u_idx, v_idx) representing edges
                                          using 0 to num_nodes-1 indices.
        delta_thresh (float): The delta threshold for the quasi-clique.
        time_limit_lp (int): Time limit in seconds for the LP solver.

    Returns:
        np.array or None: The binary x_i solution from LP if successful, else None.
    """

    if k_target < 2:
        print(f"LP Projection: k_target={k_target} is too small. Returning None.")
        return None
        
    C_k_target = math.floor(k_target * (k_target - 1) * delta_thresh / 2.0) + 1
    M_edges = len(edge_idx_pairs_numpy_list)

    num_lp_vars = num_nodes + M_edges # x_i variables + y_ij variables

    # Objective: min - (final_v_numpy @ x)  (to maximize final_v_numpy @ x)
    c_lp = np.zeros(num_lp_vars)
    c_lp[:num_nodes] = -final_v_numpy

    # Equality Constraints: A_eq @ X = b_eq
    # 1. sum_i x_i = k_target
    A_eq_data = [1.0] * num_nodes
    A_eq_row_indices = [0] * num_nodes # All contribute to the first (and only) eq constraint row
    A_eq_col_indices = list(range(num_nodes))
    A_eq = sp.csr_matrix((A_eq_data, (A_eq_row_indices, A_eq_col_indices)), shape=(1, num_lp_vars))
    b_eq = np.array([k_target])

    # Inequality Constraints: A_ub @ X <= b_ub
    # Using lists of lists for COO format before converting to CSR
    A_ub_row_constructor = []
    A_ub_col_constructor = []
    A_ub_data_constructor = []
    b_ub_vals_list = []
    current_A_ub_row_idx = 0

    # 2. Edge-Node Consistency:
    #    y_ij - x_i <= 0
    #    y_ij - x_j <= 0
    #    (Optional) x_i + x_j - y_ij <= 1  (i.e., y_ij >= x_i + x_j - 1)
    for edge_k, (u_idx, v_idx) in enumerate(edge_idx_pairs_numpy_list):
        y_k_var_idx = num_nodes + edge_k # Index of y_ij variable for this edge

        # Constraint y_ij - x_i <= 0
        A_ub_row_constructor.extend([current_A_ub_row_idx, current_A_ub_row_idx])
        A_ub_col_constructor.extend([y_k_var_idx, u_idx])
        A_ub_data_constructor.extend([1.0, -1.0])
        b_ub_vals_list.append(0.0)
        current_A_ub_row_idx += 1

        # Constraint y_ij - x_j <= 0
        A_ub_row_constructor.extend([current_A_ub_row_idx, current_A_ub_row_idx])
        A_ub_col_constructor.extend([y_k_var_idx, v_idx])
        A_ub_data_constructor.extend([1.0, -1.0])
        b_ub_vals_list.append(0.0)
        current_A_ub_row_idx += 1
        
        # Optional Constraint: x_i + x_j - y_ij <= 1  (Helps link y to sum of x for binary x)
        A_ub_row_constructor.extend([current_A_ub_row_idx, current_A_ub_row_idx, current_A_ub_row_idx])
        A_ub_col_constructor.extend([u_idx, v_idx, y_k_var_idx])
        A_ub_data_constructor.extend([1.0, 1.0, -1.0])
        b_ub_vals_list.append(1.0)
        current_A_ub_row_idx += 1


    # 3. Sufficient Edges: sum_y_ij >= C_k_target  =>  -sum_y_ij <= -C_k_target
    for edge_k in range(M_edges):
        y_k_var_idx = num_nodes + edge_k
        A_ub_row_constructor.append(current_A_ub_row_idx)
        A_ub_col_constructor.append(y_k_var_idx)
        A_ub_data_constructor.append(-1.0)
    b_ub_vals_list.append(-C_k_target)
    current_A_ub_row_idx += 1
    
    A_ub = sp.csr_matrix((A_ub_data_constructor, (A_ub_row_constructor, A_ub_col_constructor)), 
                         shape=(current_A_ub_row_idx, num_lp_vars))
    b_ub = np.array(b_ub_vals_list)
    
    bounds = [(0, 1) for _ in range(num_lp_vars)]

    print(f"Solving LP for k_target={k_target} with {num_lp_vars} vars, "
          f"{A_ub.shape[0]} UB constraints, {A_eq.shape[0]} EQ constraints. Min_edges C(k)={C_k_target}")
    
    solver_method = 'highs' 
    lp_result = None
    try:
        lp_result = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=solver_method,
                            options={"disp": True, "presolve": True, "time_limit": time_limit_lp})
    except ValueError as e:
        print(f"LP solver '{solver_method}' failed or not available: {e}. Trying 'interior-point'.")
        try:
            solver_method = 'interior-point' # Interior point can be very slow/memory hungry for this
            lp_result = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=solver_method, 
                                options={"disp": True, "presolve": True, "time_limit": time_limit_lp, "sparse": True})
        except Exception as e2:
            print(f"LP solver 'interior-point' also failed: {e2}")
            return None

    if lp_result and lp_result.success:
        print(f"LP solved successfully for k_target={k_target}. Objective achieved: {-lp_result.fun:.4f} (lower is better for original max problem). Status: {lp_result.status}")
        x_lp_solution = lp_result.x[:num_nodes]
        # Sanity check values (sometimes HiGHS can return slightly out of bounds values due to tolerances)
        x_lp_solution = np.clip(x_lp_solution, 0, 1)
        return x_lp_solution
    elif lp_result:
        print(f"LP failed or no feasible solution for k_target={k_target}. Status: {lp_result.status}, Message: {lp_result.message}")
        return None
    else: # If both solver attempts failed before linprog could return a result object
        print(f"LP solving failed catastrophically for k_target={k_target}.")
        return None

# --- Example of how you might call this after your JAX loop ---
# (Assuming your JAX script variables are in scope)

# final_v is the JAX array from sigmoid(params)
final_v_numpy = np.array(load_solution("resulting_v_2.csv"))

# edge_idx_pairs is your JAX array of edge indices
edge_idx_pairs_numpy_list = [tuple(pair) for pair in np.array(edge_idx_pairs)]

K_TARGET_TO_TRY = 200 # Or loop through a range, e.g., K_BEST_DISCRETE, K_BEST_DISCRETE+1, etc.

print(f"\nAttempting LP projection for k_target = {K_TARGET_TO_TRY}")
x_from_lp = lp_project_custom_vars(final_v_numpy, K_TARGET_TO_TRY,
                                   num_nodes, edge_idx_pairs_numpy_list,
                                   delta_thresh=0.25)

def count_edges_in_set(set_of_nodes, edges):
    return sum(1 for u, v in edges if u in set_of_nodes and v in set_of_nodes)

def check_quasi_clique(k_discrete, m_discrete, delta):
    return m_discrete > (k_discrete*(k_discrete+1)/2)*delta 

if x_from_lp is not None:
    print(f"LP solution for x (first 10): {x_from_lp[:10]}")
    print(f"Sum of x_from_lp: {np.sum(x_from_lp)} (should be close to k_target if feasible)")

    with open('result_after_lp_of_resulting_v2.csv', 'w') as f:
        f.write(','.join(map(str, x_from_lp.tolist())))

    # Now, you need to round x_from_lp to a binary set {0,1}
    # e.g., thresholding or randomized rounding, then check QC condition
    # Example: simple thresholding
    # threshold = 0.5 
    # discrete_nodes_indices = np.where(x_from_lp > threshold)[0]

    # randomized rounding (accept index i with probability x_from_lp[i])
    idxes_accepted = []
    for i in range(len(x_from_lp)):
        if np.random.rand() < x_from_lp[i]:
            idxes_accepted.append(i)
    discrete_nodes_indices = idxes_accepted
    
    # Convert compressed indices back to original node IDs if needed for submission
    selected_original_node_ids = {from_compressed_idx[i] for i in discrete_nodes_indices}
    
    # Then evaluate this selected_original_node_ids set
    k_discrete = len(selected_original_node_ids)
    m_discrete = count_edges_in_set(selected_original_node_ids, edge_idx_pairs)
    if check_quasi_clique(k_discrete, m_discrete, 0.25):
        print(f"LP-projected and rounded solution (k={k_discrete}, m={m_discrete}) IS a valid quasi-clique!")
    else:
        print(f"LP-projected and rounded solution (k={k_discrete}, m={m_discrete}) is NOT a valid quasi-clique.")
else:
    print(f"No feasible solution found by LP for k_target = {K_TARGET_TO_TRY}")