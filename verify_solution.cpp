#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <iomanip>

using namespace std;

struct Args {
    string solution_file;
    string edges_file = "edges.csv";
    int denominator = -1;  // -1 means not specified
};

void print_usage(const char* program_name) {
    cout << "Usage: " << program_name << " solution_file [options]\n"
              << "\nOptions:\n"
              << "  -e, --edges FILE     Edges file (default: edges.csv)\n"
              << "  -d, --denominator N  Denominator for threshold (1/N)\n"
              << "  -h, --help          Show this help message\n"
              << "\nExamples:\n"
              << "  " << program_name << " solution.txt\n"
              << "  " << program_name << " solution.txt -d 8\n"
              << "  " << program_name << " solution.txt --edges graph.csv --denominator 10\n";
}

Args parse_args(int argc, char* argv[]) {
    Args args;
    
    if (argc < 2) {
        print_usage(argv[0]);
        exit(1);
    }
    
    args.solution_file = argv[1];
    
    for (int i = 2; i < argc; i++) {
        string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "-e" || arg == "--edges") {
            if (i + 1 >= argc) {
                cerr << "Error: " << arg << " requires an argument\n";
                exit(1);
            }
            args.edges_file = argv[++i];
        } else if (arg == "-d" || arg == "--denominator") {
            if (i + 1 >= argc) {
                cerr << "Error: " << arg << " requires an argument\n";
                exit(1);
            }
            args.denominator = atoi(argv[++i]);
            if (args.denominator <= 0) {
                cerr << "Error: denominator must be a positive integer\n";
                exit(1);
            }
        } else {
            cerr << "Error: unknown option " << arg << "\n";
            exit(1);
        }
    }
    
    return args;
}

vector<long long> load_solution(const string& filename) {
    vector<long long> nodes;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error: Could not open solution file '" << filename << "'\n";
        exit(1);
    }
    
    string line;
    int line_num = 0;
    while (getline(file, line)) {
        line_num++;
        
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        
        if (line.empty()) continue;
        
        try {
            long long node = stoll(line);
            nodes.push_back(node);
        } catch (const exception& e) {
            cerr << "Error: Invalid node ID on line " << line_num << ": '" << line << "'\n";
            exit(1);
        }
    }
    
    return nodes;
}

unordered_map<long long, unordered_set<long long>> load_edges(const string& filename) {
    unordered_map<long long, unordered_set<long long>> edges;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error: Could not open edges file '" << filename << "'\n";
        exit(1);
    }
    
    string line;
    int line_num = 0;
    int edge_count = 0;
    
    while (getline(file, line)) {
        line_num++;
        
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        
        if (line.empty()) continue;
        
        size_t comma_pos = line.find(',');
        if (comma_pos == string::npos) {
            cerr << "Warning: Line " << line_num << " missing comma, skipping: '" << line << "'\n";
            continue;
        }
        
        try {
            string node1_str = line.substr(0, comma_pos);
            string node2_str = line.substr(comma_pos + 1);
            
            // Trim whitespace from both parts
            node1_str.erase(0, node1_str.find_first_not_of(" \t"));
            node1_str.erase(node1_str.find_last_not_of(" \t") + 1);
            node2_str.erase(0, node2_str.find_first_not_of(" \t"));
            node2_str.erase(node2_str.find_last_not_of(" \t") + 1);
            
            long long node1 = stoll(node1_str);
            long long node2 = stoll(node2_str);
            
            // Add bidirectional edges
            edges[node1].insert(node2);
            edges[node2].insert(node1);
            edge_count++;
            
        } catch (const exception& e) {
            cerr << "Warning: Invalid edge on line " << line_num << ": '" << line << "'\n";
            continue;
        }
    }
    
    cerr << "Loaded " << edge_count << " edges from " << filename << "\n";
    return edges;
}

double calculate_density(const vector<long long>& solution_nodes, 
                        const unordered_map<long long, unordered_set<long long>>& edges) {
    int n = solution_nodes.size();
    if (n < 2) {
        return 0.0;
    }
    
    // Convert solution to set for O(1) lookup
    unordered_set<long long> solution_set(solution_nodes.begin(), solution_nodes.end());
    
    int edge_count = 0;
    
    // Count edges within the solution subgraph
    for (auto node : solution_nodes) {
        auto it = edges.find(node);
        if (it != edges.end()) {
            // Count edges to other nodes in the solution
            for (auto neighbor : it->second) {
                if (solution_set.count(neighbor)) {
                    edge_count++;
                }
            }
        }
    }
    
    // Each edge is counted twice (once from each endpoint)
    edge_count /= 2;
    
    // Density = actual_edges / possible_edges
    // For a clique of size n, there are n*(n-1)/2 possible edges
    long long max_possible_edges = (long long)n * (n - 1) / 2;
    
    return max_possible_edges > 0 ? (double)edge_count / max_possible_edges : 0.0;
}

int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);
    
    // Load solution and edges
    vector<long long> solution_nodes = load_solution(args.solution_file);
    auto edges = load_edges(args.edges_file);
    
    cerr << "Solution contains " << solution_nodes.size() << " nodes\n";
    
    // Check for duplicate nodes
    unordered_set<long long> unique_nodes(solution_nodes.begin(), solution_nodes.end());
    if (unique_nodes.size() != solution_nodes.size()) {
        cerr << "Warning: Solution contains " 
                  << (solution_nodes.size() - unique_nodes.size()) 
                  << " duplicate nodes\n";
        solution_nodes.assign(unique_nodes.begin(), unique_nodes.end());
    }
    
    // Check if solution nodes exist in graph
    unordered_set<long long> nodes_in_graph;
    for (const auto& pair : edges) {
        nodes_in_graph.insert(pair.first);
        for (long long neighbor : pair.second) {
            nodes_in_graph.insert(neighbor);
        }
    }
    
    vector<long long> missing_nodes;
    for (auto node : unique_nodes) {
        if (nodes_in_graph.find(node) == nodes_in_graph.end()) {
            missing_nodes.push_back(node);
        }
    }
    
    if (!missing_nodes.empty()) {
        cerr << "Warning: " << missing_nodes.size() 
                  << " solution nodes not found in graph";
        if (missing_nodes.size() <= 10) {
            cerr << ": ";
            for (size_t i = 0; i < missing_nodes.size(); i++) {
                if (i > 0) cerr << ", ";
                cerr << missing_nodes[i];
            }
        } else {
            cerr << " (showing first 10): ";
            for (int i = 0; i < 10; i++) {
                if (i > 0) cerr << ", ";
                cerr << missing_nodes[i];
            }
            cerr << "...";
        }
        cerr << "\n";
    }
    
    // Calculate density
    double density = calculate_density(solution_nodes, edges);
    
    // Output results
    cout << fixed << setprecision(8);
    cout << "Density: " << density << "\n";
    
    if (args.denominator != -1) {
        double threshold = 1.0 / args.denominator;
        bool meets_threshold = density >= threshold;
        
        cout << "Threshold (1/" << args.denominator << "): " << threshold << "\n";
        cout << "Meets threshold: " << (meets_threshold ? "true" : "false") << "\n";
        
        return meets_threshold ? 0 : 1;
    }
    
    return 0;
}