//
//  FastNBSim.h
//
//  Created by on 2023/4/5.
//

#ifndef FastNBSim_h
#define FastNBSim_h

#include <bits/stdc++.h>

using namespace std;

bool use_signature = false;

long long n, m;
long long cur_sol_vertex = 0;
long long cur_sol_size = 0;
long long cur_sol_edges = 0;
long long operations = 0;
long long k;
// long long p = pow(2,31) - 1;
long long p = 0x7fffffff;
double ts;
double cur_sol_quasi;
double f;
double b;

vector<long long> sorted_to_actual;
map<long long, int> actual_to_sorted;

vector<vector<int>> neighbor;
vector<int> degree;
vector<int> kcore_size;
vector<pair<int,int>> sorted_vertex;
vector<int> cur_sol;
vector<int> vec_a;
vector<int> vec_b;
vector<int> vis;
vector<vector<int>> signature;



void initial_variable();
void read_graph(string filename);
void get_k_signatures(long long node);
void sort_vertex();
void iterate_vertex();
long long core_num(long long node);
long long sift_num(long long node);
double ct_score(long long v1, long long v2);
double ct_score_minhash(long long v1, long long v2);
double get_quasi(long long node);
void get_res();



#endif /* FastNBSim_h */