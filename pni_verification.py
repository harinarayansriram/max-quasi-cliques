# solution checker for https://codex.flywire.ai/app/max_clique_challenge (tweaked for usability)

import csv
import argparse

p = argparse.ArgumentParser(
        description='Verify a combined solution file'
)
p.add_argument('edges', help='Path to file containing the graph', default='edges.csv')
p.add_argument('solution', help='Path to file containing the solution')
args = p.parse_args()

with open(args.edges, 'r', encoding='utf-8') as f:
    read_edges = [row for row in csv.reader(f)]

with open(args.solution, 'r', encoding='utf-8') as f:
    read_solution = [row for row in csv.reader(f)]    

calc_score = lambda kis: sum(k / 2**i for i, k in enumerate(kis))
calc_density = lambda e, n: sum(1 for a, b in e if a in n and b in n) / (len(n)*(len(n)-1)/2)

def evaluate(rows, edges, d=12):
    kis = [0]*d
    for r in rows:
        k, δ = len(r), calc_density(edges, set(r))
        for i in range(d):
            if δ >= 1 / 2**i: kis[i] = max(kis[i], k)
        print(f'- found subgraph of size {k} with density {δ}; {kis=} {calc_score(kis)=}')
    return calc_score(kis)

s, e = read_solution, read_edges
print(f'Final score: {evaluate(s, e)}')
