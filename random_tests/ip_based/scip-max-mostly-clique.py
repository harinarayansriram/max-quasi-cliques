#!/usr/bin/env python3
import sys
import math
import argparse
from pyscipopt import Model, quicksum

def read_graph(filename):
    """Read edges from filename and return a list of edges (tuples of 64-bit integers)
       and a set of all node IDs."""
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

def build_mapping(nodes):
    """Build a mapping from original node ID to index (0..n-1)."""
    return {node: idx for idx, node in enumerate(nodes)}

def verify_solution(selected, edges, k_val):
    """Verify that the induced subgraph (given by boolean list 'selected')
       has more than half of all possible edges.
       Returns True if 4*|E(S)| > k*(k-1)."""
    if k_val < 2:
        return False
    induced_edges = 0
    for (u, v) in edges:
        if selected[u] and selected[v]:
            induced_edges += 1
    return (induced_edges * 4) > (k_val * (k_val - 1))

def main():
    parser = argparse.ArgumentParser(
        description="Find the maximum mostly-clique using SCIP via PySCIPOpt. "
                    "The graph is read from an edges CSV file (each line: u,v)."
    )
    parser.add_argument("--edges", default="edges.csv", help="Input edges file (CSV)")
    parser.add_argument("--output", default="cliquer_clique.csv", help="Output file for mostly-clique nodes")
    parser.add_argument("--timelimit", type=float, default=300.0, help="Time limit in seconds for the solver")
    args = parser.parse_args()
    
    # Read graph from file.
    edges_raw, nodes = read_graph(args.edges)
    mapping = build_mapping(nodes)
    # Map edges to indices.
    edges = []
    for (u, v) in edges_raw:
        if u in mapping and v in mapping:
            edges.append( (mapping[u], mapping[v]) )
    numNodes = len(nodes)
    numEdges = len(edges)
    print("Found {} unique nodes and {} edges.".format(numNodes, numEdges))
    
    # Create SCIP model.
    model = Model("MostlyClique")
    model.hideOutput()  # comment out if you want SCIP verbose output
    
    # Set time limit (in seconds).
    model.setParam("limits/time", args.timelimit)
    
    # Create binary variables x[i] for each vertex.
    xvars = [model.addVar(vtype="BINARY", name="x_{}".format(i)) for i in range(numNodes)]
    
    # Create binary variables y[e] for each edge.
    yvars = [model.addVar(vtype="BINARY", name="y_{}".format(e)) for e in range(numEdges)]
    
    # Create integer variable k.
    kvar = model.addVar(vtype="INTEGER", name="k", lb=0, ub=numNodes)
    
    # Constraint: k == sum_i x[i]
    model.addCons(kvar == quicksum(xvars))
    
    # For each edge, enforce: y_e <= x[u], y_e <= x[v], and y_e >= x[u] + x[v] - 1.
    for e, (u, v) in enumerate(edges):
        model.addCons(yvars[e] <= xvars[u])
        model.addCons(yvars[e] <= xvars[v])
        model.addCons(yvars[e] >= xvars[u] + xvars[v] - 1)
    
    # Add quadratic density constraint:
    # 4 * sum(yvars) - (k^2 - k) >= 1
    dens_expr = 4 * quicksum(yvars) - (kvar * kvar - kvar)
    model.addCons(dens_expr >= 1, "density")
    
    # Objective: maximize k.
    model.setObjective(kvar, "maximize")
    
    # Solve the model.
    model.optimize()
    
    sol = model.getBestSol()
    if sol is None:
        print("No solution found within the time limit.")
        sys.exit(1)
    
    # Extract solution: determine selected vertices.
    selected = [1 if model.getSolVal(sol, xvars[i]) > 0.5 else 0 for i in range(numNodes)]
    k_val = sum(selected)
    print("Found solution with clique size k =", k_val)
    
    # Verify solution as a mostly-clique.
    if verify_solution(selected, edges, k_val):
        print("Solution verified as a mostly-clique.")
    else:
        print("WARNING: The solution does NOT satisfy the mostly-clique density constraint!")
    
    # Write solution: output original node IDs for selected vertices (space-separated).
    with open(args.output, "w") as fout:
        first = True
        for i, sel in enumerate(selected):
            if sel:
                if not first:
                    fout.write(" ")
                fout.write(str(nodes[i]))
                first = False
        fout.write("\n")
    print("Solution written to", args.output)
    
if __name__ == "__main__":
    main()
