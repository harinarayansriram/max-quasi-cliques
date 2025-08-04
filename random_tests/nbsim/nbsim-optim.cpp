#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <ctime>
#include <numeric>

using namespace std;

// Graph Representation
unordered_map<long long, int> id_map;
vector<vector<int>> adj_list;
vector<int> degrees;  // Degree of each node

// Logging and Output File
ofstream log_file("fastnbsim_log.txt");
ofstream output_file("best_quasi_clique.txt");

// Load ID Mapping
void load_id_mapping(const string &filename) {
    ifstream file(filename);
    long long original_id;
    int compressed_id;
    
    while (file >> original_id >> compressed_id) {
        id_map[original_id] = compressed_id;
    }
    file.close();
}

// Load Graph
void load_graph(const string &filename, int num_nodes) {
    adj_list.resize(num_nodes);
    degrees.resize(num_nodes, 0);
    
    ifstream file(filename);
    string line;
    
    while (getline(file, line)) {
        istringstream ss(line);
        long long u, v;
        char comma;
        ss >> u >> comma >> v;
        
        if (id_map.count(u) && id_map.count(v)) {
            int uid = id_map[u], vid = id_map[v];
            adj_list[uid].push_back(vid);
            adj_list[vid].push_back(uid);
            degrees[uid]++;
            degrees[vid]++;
        }
    }
    file.close();
}

// Sort neighbors by degree (higher first)
void sort_neighbors() {
    for (auto &neighbors : adj_list) {
        sort(neighbors.begin(), neighbors.end(), [](int a, int b) {
            return degrees[a] > degrees[b];
        });
    }
}

// Compute Edge Density of a Subgraph
double edge_density(const vector<int> &nodes) {
    int edge_count = 0;
    int k = nodes.size();
    
    unordered_map<int, bool> in_set;
    for (int v : nodes) in_set[v] = true;
    
    for (int v : nodes) {
        for (int neighbor : adj_list[v]) {
            if (in_set.count(neighbor)) edge_count++;
        }
    }
    edge_count /= 2;  // Undirected graph, so each edge is counted twice
    return (2.0 * edge_count) / (k * (k - 1));
}

// Save Best Quasi-Clique to File
void save_best_clique(const vector<int> &best_clique) {
    output_file.close();
    output_file.open("best_quasi_clique.txt");
    output_file << "Size: " << best_clique.size() << "\n";
    output_file << "Nodes:\n";
    for (int node : best_clique) {
        output_file << node << "\n";
    }
    output_file.flush();
}

// Expand Quasi-Clique from Seed Node
vector<int> expand_quasi_clique(int seed, double density_threshold) {
    vector<int> clique = {seed};
    unordered_map<int, int> common_neighbors;
    
    // Count shared connections
    for (int neighbor : adj_list[seed]) {
        common_neighbors[neighbor] = 1;
    }
    
    for(int _ = 0; _ < 3; _++) {
         // Expand clique by selecting nodes with most shared neighbors
        for (int iter = 0; iter < adj_list.size(); iter++) {
            int best_node = -1, max_common = 0;
            
            for (const auto &entry : common_neighbors) {
                if (entry.second > max_common) {
                    best_node = entry.first;
                    max_common = entry.second;
                }
            }
            
            if (best_node == -1) break; // No good candidate
            
            // Add the best node and update common neighbor counts
            clique.push_back(best_node);
            common_neighbors.erase(best_node);
            
            for (int neighbor : adj_list[best_node]) {
                if (common_neighbors.count(neighbor)) {
                    common_neighbors[neighbor]++;
                }
            }
            
            if (edge_density(clique) <= density_threshold) {
                clique.pop_back(); // Revert if density drops
            }
        }
    }
   
    
    return clique;
}

// FastNBSim Algorithm
vector<int> fast_nbsim(double density_threshold) {
    vector<int> best_clique;
    time_t start_time = time(nullptr);

    // Order vertices by degree (higher first)
    vector<int> nodes(adj_list.size());
    iota(nodes.begin(), nodes.end(), 0);
    sort(nodes.begin(), nodes.end(), [](int a, int b) {
        return degrees[a] > degrees[b];
    });

    for (int i = 0; i < nodes.size(); i++) {
        int seed = nodes[i];
        
        vector<int> candidate = expand_quasi_clique(seed, density_threshold);
        if (candidate.size() > best_clique.size()) {
            best_clique = candidate;
            save_best_clique(best_clique);
            
            time_t current_time = time(nullptr);
            log_file << "[Time: " << (current_time - start_time) << "s] New best clique found! Size: " << best_clique.size() << endl;
            log_file.flush();
            cout << "New best clique found! Size: " << best_clique.size() << endl;
        }

        if (i % 1000 == 0) {
            time_t current_time = time(nullptr);
            log_file << "[Time: " << (current_time - start_time) << "s] Processed " << i << "/" << nodes.size() << " nodes. Current best size: " << best_clique.size() << endl;
            log_file.flush();
            cout << "Processed " << i << "/" << nodes.size() << " nodes. Current best size: " << best_clique.size() << endl;
        }
    }

    return best_clique;
}

// Main Execution
int main() {
    string id_map_file = "id_mapping.csv";
    string edges_file = "edges.csv";
    int num_nodes = 134181;
    
    cout << "Loading ID Mapping...\n";
    load_id_mapping(id_map_file);
    
    cout << "Loading Graph...\n";
    load_graph(edges_file, num_nodes);
    
    cout << "Sorting Neighbors...\n";
    sort_neighbors();
    
    cout << "Running FastNBSim...\n";
    double density_threshold = 0.5; // 𝛾 = 0.5 for quasi-cliques
    vector<int> quasi_clique = fast_nbsim(density_threshold);
    
    cout << "Final Best Quasi-Clique Size: " << quasi_clique.size() << endl;
    return 0;
}
