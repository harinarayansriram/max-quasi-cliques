import networkx as nx
import pandas as pd
import pulp
import time
import random

def load_graph(edge_file):
    """Load graph from edge list file."""
    print("Loading graph...")
    df = pd.read_csv(edge_file, header=None, names=['source', 'target'])
    G = nx.Graph()
    G.add_edges_from(zip(df['source'], df['target']))
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def find_dense_subgraph(G, target_size=176, time_limit=3600, seed_nodes=None):
    """
    Find a quasi-clique of exactly target_size nodes.
    A quasi-clique has more than (k choose 2)/2 edges.
    """
    print(f"Searching for a quasi-clique of size {target_size}...")
    
    # If no seed nodes provided, use highest degree nodes
    if seed_nodes is None:
        nodes_by_degree = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        seed_nodes = [n for n, _ in nodes_by_degree[:min(500, len(nodes_by_degree))]]
    
    # Create a subgraph containing seed nodes and their neighbors
    # This helps reduce the problem size
    candidate_nodes = set(seed_nodes)
    for node in seed_nodes:
        candidate_nodes.update(G.neighbors(node))
    
    # Limit to manageable size if needed
    max_nodes = min(1000, len(candidate_nodes))
    if len(candidate_nodes) > max_nodes:
        # Take some random nodes to ensure diversity
        candidate_nodes = set(list(candidate_nodes)[:max_nodes])
    
    # Create subgraph with these nodes
    subgraph = G.subgraph(candidate_nodes)
    print(f"Working with subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
    
    # Create problem
    model = pulp.LpProblem("QuasiClique", pulp.LpMaximize)
    
    # Create variables for nodes
    x = {v: pulp.LpVariable(f"x_{v}", cat='Binary') for v in subgraph.nodes()}
    
    # Create variables for edges - but only those that exist in the graph
    edge_vars = {}
    for u, v in subgraph.edges():
        edge_vars[(u, v)] = pulp.LpVariable(f"e_{u}_{v}", cat='Binary')
        
        # Edge variables are forced by node variables
        model += edge_vars[(u, v)] <= x[u]
        model += edge_vars[(u, v)] <= x[v]
        model += edge_vars[(u, v)] >= x[u] + x[v] - 1
    
    # SIMPLIFICATION: We fix the size of the solution
    model += pulp.lpSum(x.values()) == target_size
    
    # Objective: maximize the number of edges
    model += pulp.lpSum(edge_vars.values())
    
    # Quasi-clique constraint: number of edges > (k choose 2)/2
    min_edges = (target_size * (target_size - 1)) / 4 + 0.1  # Adding 0.1 for "strictly more"
    model += pulp.lpSum(edge_vars.values()) >= min_edges
    
    # Set time limit
    if 'GUROBI' in pulp.listSolvers(onlyAvailable=True) and False:
        solver = pulp.GUROBI(timeLimit=time_limit)
    else:
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=True)
    
    # Solve
    start_time = time.time()
    model.solve(solver)
    solve_time = time.time() - start_time
    
    print(f"Solved in {solve_time:.2f} seconds. Status: {pulp.LpStatus[model.status]}")
    
    if model.status == pulp.LpStatusOptimal:
        selected_nodes = [v for v in subgraph.nodes() if x[v].value() > 0.5]
        result_subgraph = subgraph.subgraph(selected_nodes)
        
        # Verify the solution
        edge_count = result_subgraph.number_of_edges()
        possible_edges = len(selected_nodes) * (len(selected_nodes) - 1) / 2
        density = edge_count / possible_edges
        
        print(f"Found solution with {len(selected_nodes)} nodes")
        print(f"Edges: {edge_count}/{possible_edges} ({density:.4f})")
        print(f"Is quasi-clique: {edge_count > possible_edges/2}")
        
        return selected_nodes, density
    else:
        print(f"No solution found. Status: {pulp.LpStatus[model.status]}")
        return [], 0

def binary_search_max_quasi_clique(G, min_size=175, max_size=250, time_limit=600):
    """Use binary search to find the maximum size quasi-clique."""
    low = min_size
    high = max_size
    best_solution = []
    best_density = 0
    
    # First, verify we can find a solution at min_size
    # initial_solution, initial_density = find_dense_subgraph(G, target_size=min_size, time_limit=time_limit)
    initial_solution = [720575940628077568,720575940628954112,720575940628914692,720575940623788040,720575940627480586,720575940609454603,720575940628181520,720575940632403986,720575940621256212,720575940627927573,720575940644238871,720575940622368792,720575940633212441,720575940633483807,720575940603985952,720575940617537570,720575940618757666,720575940644362788,720575940631586858,720575940632416298,720575940630571052,720575940631105580,720575940610669614,720575940630945335,720575940628200506,720575940626143806,720575940636873791,720575940638633535,720575940611079236,720575940617586757,720575940608199748,720575940625251401,720575940631899223,720575940621298270,720575940620189790,720575940622054497,720575940631287912,720575940627772009,720575940616169578,720575940630546540,720575940612907121,720575940635082871,720575940610157688,720575940627844734,720575940660217473,720575940659611777,720575940648650884,720575940619902598,720575940624206472,720575940617668747,720575940631423116,720575940636241550,720575940637666446,720575940637633678,720575940634034839,720575940612284569,720575940618295454,720575940615594658,720575940622038182,720575940613579943,720575940611372714,720575940637526190,720575940613203119,720575940606751922,720575940611022515,720575940616079035,720575940635468991,720575940610408643,720575940631630021,720575940629011655,720575940619411147,720575940612475598,720575940614944977,720575940618385620,720575940608241365,720575940608845525,720575940618308825,720575940641593051,720575940623636701,720575940615303391,720575940614727903,720575940604431584,720575940623875819,720575940623444715,720575940621398254,720575940620467438,720575940620465904,720575940627035388,720575940627160322,720575940628914436,720575940628524292,720575940627690246,720575940616824588,720575940627708688,720575940626624275,720575940627605779,720575940627737365,720575940623000858,720575940624450342,720575940624547622,720575940624879911,720575940624001321,720575940632197930,720575940632198442,720575940617021234,720575940615073587,720575940644961588,720575940638778165,720575940625431866,720575940617179451,720575940619337536,720575940620565313,720575940640314190,720575940631619921,720575940628343634,720575940619895125,720575940625784153,720575940638128474,720575940620437339,720575940631328104,720575940619928429,720575940620595053,720575940611379569,720575940630183286,720575940637056887,720575940631497078,720575940630024566,720575940630770042,720575940618561403,720575940661114753,720575940648669060,720575940622920074,720575940632795532,720575940641606541,720575940628316047,720575940617207185,720575940613809554,720575940612461461,720575940628467611,720575940613167007,720575940644704160,720575940656090017,720575940615394719,720575940614238114,720575940633938349,720575940608124338,720575940608362418,720575940633804212,720575940617205179,720575940623287228,720575940635945919,720575940635933119,720575940634939327,720575940606056905,720575940631975884,720575940623136717,720575940631930828,720575940611352014,720575940630131663,720575940613489617,720575940611671506,720575940611944914,720575940610505170,720575940610048981,720575940608970197,720575940641592795,720575940623528925,720575940613590501,720575940628739560,720575940614429165,720575940619931119,720575940624515571,720575940640830453,720575940624057853,720575940623901695,]
    initial_density = 0.5023
    if not initial_solution:
        print(f"Couldn't find initial solution at size {min_size}. Trying with more time...")
        initial_solution, initial_density = find_dense_subgraph(G, target_size=min_size, time_limit=time_limit*2)
        if not initial_solution:
            print("Failed to find even initial solution. Exiting.")
            return [], 0
    
    best_solution = initial_solution
    best_density = initial_density
    best_size = min_size
    
    print(f"Found initial solution of size {min_size}. Starting binary search...")
    
    # Try incremental approach first
    current_size = min_size + 1
    while current_size <= high:
        print(f"Trying size {current_size}...")
        solution, density = find_dense_subgraph(G, target_size=current_size, time_limit=time_limit, 
                                                seed_nodes=best_solution)
        
        if solution:
            best_solution = solution
            best_density = density
            best_size = current_size
            print(f"Found new best solution of size {best_size}")
            current_size += 1
        else:
            print(f"Could not find solution at size {current_size}")
            break
    
    print(f"\nBest solution found:")
    print(f"Size: {best_size}")
    print(f"Density: {best_density:.4f}")
    
    return best_solution, best_density

def is_quasi_clique(G, nodes):
    """Check if the given nodes form a quasi-clique."""
    subgraph = G.subgraph(nodes)
    edge_count = subgraph.number_of_edges()
    possible_edges = len(nodes) * (len(nodes) - 1) / 2
    return edge_count > possible_edges / 2, edge_count / possible_edges

def main():
    # Load the graph
    G = load_graph("edges.csv")
    
    # Try to find maximum quasi-clique
    solution, density = binary_search_max_quasi_clique(G, min_size=175, max_size=250, time_limit=1800)
    
    if solution:
        is_valid, final_density = is_quasi_clique(G, solution)
        print("\nFinal verification:")
        print(f"Size: {len(solution)}")
        print(f"Density: {final_density:.4f}")
        print(f"Is valid quasi-clique: {is_valid}")
        
        # Save solution
        with open("quasi_clique_solution.txt", "w") as f:
            for node in solution:
                f.write(f"{node}\n")
    else:
        print("No solution found within the constraints.")

if __name__ == "__main__":
    main()