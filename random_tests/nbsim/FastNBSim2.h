//
//  FastNBSim.h
//
//  Created by on 2023/4/5.
//

#ifndef FastNBSim_h
#define FastNBSim_h

#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <algorithm>
#include <unordered_set>
#include <iomanip>


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


vector<vector<long long>> neighbor;
vector<long long> degree;
vector<long long> kcore_size;
vector<pair<long long,long long>> sorted_vertex;
vector<long long> cur_sol;
vector<long long> vec_a;
vector<long long> vec_b;
vector<long long> vis;
vector<vector<long long>> signature;



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