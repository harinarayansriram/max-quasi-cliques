#!/usr/bin/env python3

from random import random, randint

# Start out with some current solution
# Add node that improves density the most (may be negative)
# Keep track of how many iterations we've been below the threshold
# Prune worst nodes to get back above the threshold

DENSITY_THRESHOLD = 1.0/8.0

def read_edges(filename):
    """
    Read edges from a CSV file where each line is 'u,v' (each u and v are integers).
    Returns a list of edges (tuples) and a set of all node IDs.
    """
    edges = []
    nodes = set()
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 2:
                continue
            try:
                u = int(parts[0])
                v = int(parts[1])
            except ValueError:
                continue
            edges.append((u, v))
            nodes.add(u)
            nodes.add(v)
    return edges, sorted(list(nodes))

def read_solution(filename):
    """
    Read the solution file (a single line of space-separated node IDs)
    and return a set of integers.
    """
    with open(filename, "r") as f:
        line = f.read().strip()
        if not line:
            return set()
        parts = line.split()
        try:
            sol_nodes = {int(x) for x in parts}
        except ValueError:
            sol_nodes = set()
    return sol_nodes

def build_adjacency_dict(edges):
    """
    Build an adjacency dictionary mapping each node to a set of its neighbors.
    """
    adj = {}
    for u, v in edges:
        if u not in adj:
            adj[u] = set()
        if v not in adj:
            adj[v] = set()
        adj[u].add(v)
        adj[v].add(u)
    return adj

def calculate_density(sol_nodes, adj):
    induced_edges = 0
    for v in sol_nodes:
        # Count neighbors in the solution
        neighbors_in_S = adj.get(v, set()).intersection(sol_nodes)
        local_degree = len(neighbors_in_S)
        induced_edges += local_degree
    
    induced_edges //= 2
    return induced_edges/( len(sol_nodes) * (len(sol_nodes) - 1)/2 )

def add_best_node(nodes, curr_edges, density_change_by_adding_node: dict, sol_nodes: set, adj):
    if not density_change_by_adding_node:
        # density_change_by_adding_node = {}
        for node in nodes:
            if node in sol_nodes:
                continue
            density_change_by_adding_node[node] = 0
            for neighbor in adj[node]:
                if neighbor in sol_nodes:
                    density_change_by_adding_node[node] += 1
    
    best_node_to_add = max(density_change_by_adding_node, key=lambda x: (density_change_by_adding_node.get(x), len(adj[x]), random() ))
    sol_nodes.add(best_node_to_add)
    new_edges = density_change_by_adding_node[best_node_to_add] + curr_edges
    new_density = new_edges / ( len(sol_nodes) * (len(sol_nodes) - 1)/2 )

    # update density change by adding node based on the node we just added
    # density_change_by_adding_node[best_node_to_add] = 0
    density_change_by_adding_node.pop(best_node_to_add)
    for neighbor in adj[best_node_to_add]:
        if neighbor not in sol_nodes:
            density_change_by_adding_node[neighbor] += 1

    return (best_node_to_add, new_edges, len(sol_nodes), new_density)

def remove_worst_node(nodes, curr_edges, density_change_by_adding_node: dict, sol_nodes: set, adj):
    if not density_change_by_adding_node:
        # density_change_by_adding_node = {}
        for node in nodes:
            if node in sol_nodes:
                continue
            density_change_by_adding_node[node] = 0
            for neighbor in adj[node]:
                if neighbor in sol_nodes:
                    density_change_by_adding_node[node] += 1
    
    density_change_by_removing_node = {}
    for node in sol_nodes:
        density_change_by_removing_node[node] = 0
        for neighbor in adj[node]:
            if neighbor in sol_nodes:
                density_change_by_removing_node[node] += 1

    worst_node_to_remove = min(density_change_by_removing_node, key=lambda x: (density_change_by_adding_node.get(x), random() ))
    # worst_node_to_remove = min({k:density_change_by_adding_node[k] for k in sol_nodes}, key=density_change_by_adding_node.get)
    sol_nodes.remove(worst_node_to_remove)
    new_edges = curr_edges - density_change_by_removing_node[worst_node_to_remove]
    new_density = new_edges / ( len(sol_nodes) * (len(sol_nodes) - 1)/2 )

    # update density change by adding node based on the node we just removed
    # density_change_by_adding_node[worst_node_to_remove] = 0
    density_change_by_adding_node[worst_node_to_remove] = 0
    for neighbor in adj[worst_node_to_remove]:
        if neighbor not in sol_nodes:
            density_change_by_adding_node[neighbor] -= 1
        else:
            density_change_by_adding_node[worst_node_to_remove] += 1

    return (worst_node_to_remove, new_edges, len(sol_nodes), new_density)

def main():
    edges, nodes = read_edges("edges.csv")

    print(f"Number of nodes: {len(nodes)}")
    # sol_nodes = read_solution("369.csv")
    sol_nodes = {720575940606246334,720575940626986885}
    adj = build_adjacency_dict(edges)

    curr_density = calculate_density(sol_nodes, adj)
    curr_edges = curr_density * ( len(sol_nodes) * (len(sol_nodes) - 1)/2 )

    

    print(f"Initial density: {curr_density:.4f} ({curr_edges} edges)")
    
    # density_change_by_adding_node = {}
    # for node in nodes:
    #     if node in sol_nodes:
    #         continue
    #     density_change_by_adding_node[node] = 0
    #     for neighbor in adj[node]:
    #         if neighbor in sol_nodes:
    #             density_change_by_adding_node[node] += 1
    # best_node_to_add = max(density_change_by_adding_node, key=density_change_by_adding_node.get)
    # print("Best node to add: {}".format(best_node_to_add))

    # sol_nodes.add(best_node_to_add)
    # expected_new_edges = density_change_by_adding_node[best_node_to_add] + curr_edges
    # expected_new_density = expected_new_edges / ( len(sol_nodes) * (len(sol_nodes) - 1)/2 )
    # print(f"Expected new density: {expected_new_density:.4f} ({expected_new_edges} edges)")

    # new_density = calculate_density(sol_nodes, adj)
    # new_edges = new_density * ( len(sol_nodes) * (len(sol_nodes) - 1)/2 )
    # print(f"New density: {new_density:.4f} ({new_edges} edges)")

    density_change_by_adding_node = {}
    

    curr_distance_from_initial = 0
    curr_distance_from_best = 0


    for _ in range(600):
        add_best_node(nodes, curr_edges, density_change_by_adding_node, sol_nodes, adj)

    while calculate_density(sol_nodes, adj) <= DENSITY_THRESHOLD:
        remove_worst_node(nodes, curr_edges, density_change_by_adding_node, sol_nodes, adj)

    print(f"Num nodes in solution: ", len(sol_nodes))
    curr_density = calculate_density(sol_nodes, adj)
    curr_edges = curr_density * len(sol_nodes) * (len(sol_nodes) - 1)/2 
    initial_num_nodes_in_solution = len(sol_nodes)
    num_nodes_in_best_solution = initial_num_nodes_in_solution
    running_best_solution = set(sol_nodes)
    running_best_density = calculate_density(sol_nodes, adj)
    running_best_edges = running_best_density * ( len(sol_nodes) * (len(sol_nodes) - 1)/2 )

    for _ in range(100000):
        print(f'Iteration: {_} of 100000, distance from initial: {curr_distance_from_initial}, distance from best: {curr_distance_from_best}. best density: {running_best_density:.4f} ({running_best_edges} edges, {num_nodes_in_best_solution} nodes)')
        amount = randint(-40, 40)
        # if random() < 0.5:
        #     amount *= -1
        
        for _ in range(abs((len(sol_nodes) - num_nodes_in_best_solution) - amount)):
            if amount > len(sol_nodes) - num_nodes_in_best_solution:
                print(add_best_node(nodes, curr_edges, density_change_by_adding_node, sol_nodes, adj))
                new_density = calculate_density(sol_nodes, adj)
                if len(sol_nodes) == num_nodes_in_best_solution and new_density > running_best_density:
                    running_best_solution = set(sol_nodes)
                    running_best_density = new_density
                    running_best_edges = new_density * ( len(sol_nodes) * (len(sol_nodes) - 1)/2 )
                    print("Better density for same size at ", len(sol_nodes), "of", new_density)
                    curr_distance_from_best = 0
                elif len(sol_nodes) > num_nodes_in_best_solution and new_density > DENSITY_THRESHOLD:
                    print("Improvement!!!", len(sol_nodes), new_density)
                    print(list(sol_nodes))

                    running_best_solution = set(sol_nodes)

                    running_best_density = new_density
                    running_best_edges = new_density * ( len(sol_nodes) * (len(sol_nodes) - 1)/2 )

                    num_nodes_in_best_solution = len(sol_nodes)

                    curr_distance_from_best = 0
                else:
                    curr_distance_from_best = len(sol_nodes)-amount

                    # raise Exception("Found better solution!!")

                    
            else:
                print(remove_worst_node(nodes, curr_edges, density_change_by_adding_node, sol_nodes, adj))
                new_density = calculate_density(sol_nodes, adj)

                if len(sol_nodes) == num_nodes_in_best_solution and new_density > running_best_density:
                    running_best_solution = set(sol_nodes)
                    running_best_density = new_density
                    running_best_edges = new_density * ( len(sol_nodes) * (len(sol_nodes) - 1)/2 )
                    print("Better density for same size at ", len(sol_nodes), "of", new_density)
                    curr_distance_from_best = 0
                elif len(sol_nodes) > num_nodes_in_best_solution and new_density > DENSITY_THRESHOLD:
                    print("Improvement!!!", len(sol_nodes), new_density)
                    print(list(sol_nodes))

                    running_best_solution = set(sol_nodes)

                    running_best_density = new_density
                    running_best_edges = new_density * ( len(sol_nodes) * (len(sol_nodes) - 1)/2 )

                    num_nodes_in_best_solution = len(sol_nodes)

                    curr_distance_from_best = 0
                else:
                    curr_distance_from_best = len(sol_nodes)-amount

                    # raise Exception("Found better solution!!")

        curr_distance_from_initial = len(sol_nodes) - initial_num_nodes_in_solution  
        curr_distance_from_best = len(sol_nodes) - num_nodes_in_best_solution    

        print(f"Current distance from initial: {curr_distance_from_initial}, distance from best: {curr_distance_from_best}. best density: {running_best_density:.4f} ({running_best_edges} edges)")


    # for _ in range(5):
    #     print(add_best_node(nodes, curr_edges, density_change_by_adding_node, sol_nodes, adj))
    #     print(calculate_density(sol_nodes, adj))
    # for _ in range(15):
    #     print(remove_worst_node(nodes, curr_edges, density_change_by_adding_node, sol_nodes, adj))
    #     print(calculate_density(sol_nodes, adj))

    # for _ in range(20):
    #     print(add_best_node(nodes, curr_edges, density_change_by_adding_node, sol_nodes, adj))
    #     print(calculate_density(sol_nodes, adj))
        
    # for _ in range(19):
    #     print(remove_worst_node(nodes, curr_edges, density_change_by_adding_node, sol_nodes, adj))
    #     print(calculate_density(sol_nodes, adj))

    # for _ in range(10):
    #     print(add_best_node(nodes, curr_edges, density_change_by_adding_node, sol_nodes, adj))
    #     print(calculate_density(sol_nodes, adj))
    
    # for _ in range(15):
    #     print(remove_worst_node(nodes, curr_edges, density_change_by_adding_node, sol_nodes, adj))
    #     print(calculate_density(sol_nodes, adj))

    # for _ in range(15):
    #     print(add_best_node(nodes, curr_edges, density_change_by_adding_node, sol_nodes, adj))
    #     print(calculate_density(sol_nodes, adj))

    # print(f"Final density: {calculate_density(sol_nodes, adj):.4f} ({len(sol_nodes) * (len(sol_nodes) - 1)/2} edges)")
    # print(f"Final num nodes in solution: ", len(sol_nodes))



if __name__ == "__main__":
    main()