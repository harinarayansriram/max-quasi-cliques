#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <random>
#include <bitset>

using namespace std;

const int NUM_HASHES = 32; // Number of hash functions for MinHash

// Graph Data Structures
unordered_map<long long, int> id_map;  // Map original node ID -> compressed ID
vector<vector<int>> adj_list;          // Adjacency list
vector<vector<int>> minhash_signatures; // MinHash signatures for each node

// MinHash hash functions
vector<int> hash_coeff_a;
vector<int> hash_coeff_b;
const int PRIME = 10000019; // Large prime for hashing

// Function to split a string (CSV parsing)
vector<string> split(const string &s, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Load ID Mapping
void load_id_mapping(const string &filename) {
    ifstream file(filename);
    string line;
    int compressed_id;
    long long original_id;
    
    while (getline(file, line)) {
        istringstream iss(line);
        iss >> original_id >> compressed_id;
        id_map[original_id] = compressed_id;
    }
    
    file.close();
}

// Load Graph Edges
void load_graph(const string &filename, int num_nodes) {
    adj_list.resize(num_nodes);
    
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        auto tokens = split(line, ',');
        long long u = stoll(tokens[0]);
        long long v = stoll(tokens[1]);

        if (id_map.count(u) && id_map.count(v)) {
            int uid = id_map[u];
            int vid = id_map[v];

            adj_list[uid].push_back(vid);
            adj_list[vid].push_back(uid);
        }
    }
    
    file.close();
}

// Initialize MinHash Functions
void init_minhash_functions(int num_hashes, int max_val) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(1, max_val - 1);

    for (int i = 0; i < num_hashes; i++) {
        hash_coeff_a.push_back(dist(gen));
        hash_coeff_b.push_back(dist(gen));
    }
}

// Compute MinHash Signature for a Node
vector<int> compute_minhash(int node, int num_hashes) {
    vector<int> signature(num_hashes, PRIME);
    
    for (int neighbor : adj_list[node]) {
        for (int i = 0; i < num_hashes; i++) {
            int hash_val = (hash_coeff_a[i] * neighbor + hash_coeff_b[i]) % PRIME;
            signature[i] = min(signature[i], hash_val);
        }
    }
    
    return signature;
}

// Build MinHash Signatures for All Nodes
void build_minhash_signatures(int num_nodes, int num_hashes) {
    minhash_signatures.resize(num_nodes);
    
    for (int i = 0; i < num_nodes; i++) {
        minhash_signatures[i] = compute_minhash(i, num_hashes);
    }
}

// Compute MinHash Similarity between Two Nodes
double minhash_similarity(int u, int v) {
    int match_count = 0;
    
    for (int i = 0; i < NUM_HASHES; i++) {
        if (minhash_signatures[u][i] == minhash_signatures[v][i]) {
            match_count++;
        }
    }
    
    return static_cast<double>(match_count) / NUM_HASHES;
}

// Extract Large Quasi-Clique
vector<int> extract_quasi_clique(int start_node, double gamma, double beta) {
    vector<int> clique;
    
    for (int neighbor : adj_list[start_node]) {
        if (minhash_similarity(start_node, neighbor) >= gamma) {
            clique.push_back(neighbor);
        }
    }
    
    if (clique.size() < beta * adj_list[start_node].size()) {
        return {};
    }
    
    return clique;
}

// FastNBSim Algorithm
vector<int> fast_nbsim(double gamma, double beta) {
    vector<int> best_clique;
    
    for (int i = 0; i < adj_list.size(); i++) {
        vector<int> candidate_clique = extract_quasi_clique(i, gamma, beta);
        if (candidate_clique.size() > best_clique.size()) {
            best_clique = candidate_clique;
        }
    }
    
    return best_clique;
}

int main() {
    string id_map_file = "id_mapping.csv";
    string edges_file = "edges.csv";
    int num_nodes = 134181;
    
    cout << "Loading ID Mapping...\n";
    load_id_mapping(id_map_file);
    
    cout << "Loading Graph...\n";
    load_graph(edges_file, num_nodes);
    
    cout << "Initializing MinHash Functions...\n";
    init_minhash_functions(NUM_HASHES, num_nodes);
    
    cout << "Computing MinHash Signatures...\n";
    build_minhash_signatures(num_nodes, NUM_HASHES);
    
    cout << "Running FastNBSim...\n";
    double gamma = 0.5;
    double beta = 0.75;
    vector<int> quasi_clique = fast_nbsim(gamma, beta);
    
    cout << "Found Quasi-Clique of size: " << quasi_clique.size() << endl;
    return 0;
}
