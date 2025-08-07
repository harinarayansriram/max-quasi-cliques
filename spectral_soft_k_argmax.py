import time
import numpy as np

import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from functools import partial

import jax
import jax.numpy as jnp
from jax import grad
import jax.experimental.sparse as jsparse

find_tau_iters = 20

@jax.custom_jvp
def find_tau(x: jnp.ndarray,
             k: int,
             eps: float = 1e-2 ) -> jnp.ndarray:
    """
    Solves for tau in: sum_i sigmoid((x_i - tau)/eps) == k
    via bisection. Returns tau (scalar).
    """
    lo = jnp.min(x) - 1.0
    hi = jnp.max(x) + 1.0

    def body(state, _):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        s = jax.nn.sigmoid((x - mid) / eps)
        lo, hi = jax.lax.cond(
            jnp.sum(s) > k,
            lambda _: (mid, hi),
            lambda _: (lo, mid),
            operand=None,
        )
        return (lo, hi), None

    (lo, hi), _ = jax.lax.scan(body, (lo, hi), None, length=find_tau_iters)
    return 0.5 * (lo + hi)

@find_tau.defjvp
def find_tau_jvp(primals, tangents):
    x, k, eps = primals
    x_dot, *_ = tangents
    tau = find_tau(x, k, eps)
    diff = (x - tau) / eps
    sigma = jax.nn.sigmoid(diff)
    sigma_p = sigma * (1 - sigma) / eps
    g_tau = -jnp.sum(sigma_p)
    g_x = sigma_p
    tau_dot = - jnp.vdot(g_x, x_dot) / g_tau
    return tau, tau_dot


def soft_k_mask(x: jnp.ndarray,
                k: int,
                eps: float = 1e-2) -> jnp.ndarray:
    """
    Returns a soft k-hot mask vector s in [0,1]^n summing to k.
    As eps->0, s converges to the indicator of the top-k indices.
    """
    tau = find_tau(x, k, eps)
    return jax.nn.sigmoid((x - tau) / eps)

@partial(jax.jit, static_argnames=('max_iters','tol','k'))
def _optimize_core(
    A_sp: jsparse.BCOO,
    k: int,
    x0: jnp.ndarray,
    eta: float,
    max_iters: int,
    tol: float,
    eps: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Gradient ascent on f(x)=s.T @ A_sp @ s, starting at x0.
    """
    grad_obj = grad(lambda x: soft_k_mask(x, k, eps).T @ (A_sp @ soft_k_mask(x, k, eps)))
    def cond_fun(state):
        i, x = state
        g = grad_obj(x)
        x_new = x + eta * g
        return (i < max_iters) & (jnp.linalg.norm(x_new - x) >= tol)
    def body_fun(state):
        i, x = state
        g = grad_obj(x)
        x_new = x + eta * g
        return (i+1, x_new)
    _, x_final = jax.lax.while_loop(cond_fun, body_fun, (0, x0))

    s_cont = soft_k_mask(x_final, k, eps)
    _, idxs = jax.lax.top_k(s_cont, k)
    X_hard = jnp.zeros(s_cont.shape[0], dtype=bool).at[idxs].set(True)
    return X_hard, s_cont

def optimize_continuous(
    A,
    k: int,
    x_init: jnp.ndarray = None,
    eta: float = 5e-4,
    max_iters: int = 500,
    tol: float = 1e-6,
    eps: float = 1e-2,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    
    if isinstance(A, jsparse.BCOO):
        A_sp = A
    else:
        A_sp = jsparse.BCOO.from_scipy_sparse(A)

    n = A_sp.shape[0]
    # init with leading eigenvector if none provided
    if x_init is None:
        _, v = eigsh(A, k=1, which='LA', tol=1e-6, maxiter=1000)
        x0 = jnp.array(v.ravel())
    else:
        x0 = x_init
    # normalize
    x0 = jnp.asarray(x0)
    x0 /= jnp.linalg.norm(x0)

    X_hard_jax, s_cont_jax = _optimize_core(A_sp, k, x0, eta, max_iters, tol, eps)
    # move to CPU and convert to numpy
    X_hard = jnp.asarray(X_hard_jax).astype(bool)
    s_cont = jnp.asarray(s_cont_jax)
    return jax.device_get(X_hard), jax.device_get(s_cont)

def binary_search_in_range(A, threshold, low, high):
    best_k = None
    best_result = None

    while low <= high:
        mid = (low + high) // 2
        X_mid, z_mid = optimize_continuous(A, mid)
        mid_density = (X_mid.T @ A @ X_mid) / (mid * (mid - 1))
        print(f'  Binary search: K={mid}, density={mid_density:.6f}')

        if mid_density >= threshold:
            best_k = mid
            best_result = (mid_density, X_mid, z_mid)
            low = mid + 1
        else:
            high = mid - 1

    if best_k is not None:
        return best_k, best_result[0], best_result[1], best_result[2] # (optimal_k, density, X, z)
    return None



adaptive_step_cap = 512
def find_optimal_k_adaptive(A, threshold, initial_k, max_k=None):
    n = A.shape[0]
    if max_k is None:
        max_k = n - 1

    print(f'Starting adaptive search for optimal K with threshold {threshold}')

    # test initial k
    X_curr, z_curr = optimize_continuous(A, initial_k)
    curr_density = (X_curr.T @ A @ X_curr) / (initial_k * (initial_k - 1))
    print(f'Initial K={initial_k}, density={curr_density:.6f}')

    if curr_density >= threshold:
        current_k = initial_k
        step_size = 1
        last_good_k = current_k

        while current_k + step_size <= max_k:
            test_k = current_k + step_size
            X_test, z_test = optimize_continuous(A, test_k)
            test_density = (X_test.T @ A @ X_test) / (test_k * (test_k - 1))
            print(f'Exponential search: K={test_k}, density={test_density:.6f}, step={step_size}')

            if test_density >= threshold:
                last_good_k = test_k
                current_k = test_k
                step_size = min(step_size * 2, adaptive_step_cap)  # exponential growth, capped
            else:
                # went too far, binary search between last_good_k and test_k
                print(f'Switching to binary search between K={last_good_k} and K={test_k}')
                result = binary_search_in_range(A, threshold, last_good_k + 1, test_k - 1)
                if result is not None:
                    final_k, final_density, final_X, final_z = result
                    # check if we found something better than last_good_k
                    if final_k > last_good_k:
                        last_good_k = final_k
                break

        # get final result for last_good_k
        X_final, z_final = optimize_continuous(A, last_good_k)
        final_density = (X_final.T @ A @ X_final) / (last_good_k * (last_good_k - 1))

    else:
        # density too low, use exponential steps downward then binary search
        current_k = initial_k
        step_size = 1
        lower_bound = 2  

        while current_k - step_size >= lower_bound:
            test_k = current_k - step_size
            X_test, z_test = optimize_continuous(A, test_k)
            test_density = (X_test.T @ A @ X_test) / (test_k * (test_k - 1))
            print(f'Exponential search down: K={test_k}, density={test_density:.6f}, step={step_size}')

            if test_density >= threshold:
                # found a good region, binary search between test_k and current_k
                print(f'Switching to binary search between K={test_k} and K={current_k}')
                result = binary_search_in_range(A, threshold, test_k, current_k - 1)
                if result is not None:
                    last_good_k, final_density, X_final, z_final = result
                    break
                
                last_good_k = test_k
                final_density = test_density
                X_final = X_test
                z_final = z_test
                break
            else:
                current_k = test_k
                step_size = min(step_size * 2, adaptive_step_cap)
        else:
            # exhausted downward search, do final binary search
            print(f'Final binary search between K={lower_bound} and K={current_k}')
            result = binary_search_in_range(A, threshold, lower_bound, current_k - 1)
            if result is not None:
                last_good_k, final_density, X_final, z_final = result
            else:
                raise ValueError(f'No K >= {lower_bound} satisfies density threshold {threshold}')

    print(f'\nOptimal K found: {last_good_k}')
    print(f'Final density: {final_density:.6f}')
    print(f'Threshold: {threshold}')

    return last_good_k, final_density, X_final, z_final

if __name__ == '__main__':
    id_mapping = {}
    rev_mapping = []
    with open('id_mapping.csv', 'r') as f:
        for line in f:
            id1, idx = map(int, line.strip().split(' '))
            id_mapping[id1] = idx
            rev_mapping.append(id1)

    sparse_matrix = sp.lil_matrix((len(id_mapping), len(id_mapping)))
    with open('edges.csv', 'r') as f:
        for line in f:
            id1, id2 = map(int, line.strip().split(','))
            sparse_matrix[id_mapping[id1], id_mapping[id2]] = 1
            sparse_matrix[id_mapping[id2], id_mapping[id1]] = 1

    A = sparse_matrix.tocsr()
    print('Loaded matrix, starting optimization')

    denoms = [2**k for k in range(1,12)]
    init_ks = [127, 298, 623, 1447, 2722, 5690, 10704, 20139, 34106, 57760, 57760]
    
    summary_stats = []

    for denom, init_k in zip(denoms, init_ks):
        print('\n\n\n','-'*50)
        threshold = 1.0/denom
        initial_k = init_k
        start_time = time.time()

        print(f'Running with threshold: {threshold} (denom: {denom}), initial K: {initial_k}')
        optimal_k, final_density, X_final, z_final = find_optimal_k_adaptive(
            A, threshold, initial_k
        )

        end_time = time.time()

        print(f'Time taken: {end_time - start_time:.2f} seconds')
        print(f'\nFinal Results:')
        print(f'Optimal K: {optimal_k}')
        print(f'Final Density: {final_density:.6f}')
        print(f'Non-zero X_final: {(X_final != 0).sum()}')

        summary_stats.append((denom, initial_k, optimal_k, final_density, end_time - start_time))

        # get the actual top k indices
        topK_idx = jnp.where(X_final > 0)[0]

        # print(f'Top {optimal_k} elements: {topK_idx.tolist()}')
        topK_elements = [rev_mapping[idx] for idx in topK_idx]

        with open(f'run_1/{denom}_result.csv', 'w') as f:
            f.write('\n'.join(map(str, topK_elements)))

    with open('run_1/summary_stats.csv', 'w') as f:
        f.write('denom,initial_k,optimal_k,final_density,time_taken')
        for denom, init_k, optimal_k, final_density, time_taken in summary_stats:
            f.write(f'\n{denom},{init_k},{optimal_k},{final_density},{time_taken}')

