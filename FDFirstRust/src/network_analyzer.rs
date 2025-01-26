use crate::params;
use std::collections::VecDeque;
use std::{f64, usize};

pub struct NetworkAnalyzer {
    pub shortest_path: Vec<Vec<isize>>, // declared as an signed integer to allow for -1 as a procedural marker.

    adj_list: Vec<Vec<usize>>,

    pub average_path_length: f64,
    pub network_efficiency: f64,
    pub global_clustering_watts_strogatz: f64,
    pub centralization_closeness: f64,
    pub centralization_triadic_participation: f64,
    pub shortest_path_variance: f64,
}

impl NetworkAnalyzer {
    // ----------------------------------------------------------------
    // GETTERS (equivalent to Java getAveragePathLength(), etc.)
    // ----------------------------------------------------------------
    pub fn get_average_path_length(&self) -> f64 {
        self.average_path_length
    }

    pub fn get_network_efficiency(&self) -> f64 {
        self.network_efficiency
    }

    pub fn get_global_clustering_watts_strogatz(&self) -> f64 {
        self.global_clustering_watts_strogatz
    }

    pub fn get_closeness_centralization(&self) -> f64 {
        self.centralization_closeness
    }

    pub fn get_triadic_centralization(&self) -> f64 {
        self.centralization_triadic_participation
    }

    pub fn get_shortest_path_variance(&self) -> f64 {
        self.shortest_path_variance
    }
    /// Initializes empty fields.
    pub fn new() -> Self {
        NetworkAnalyzer {
            shortest_path: vec![vec![-1; params::N]; params::N],
            adj_list: vec![Vec::new(); params::N],
            average_path_length: 0.0,
            network_efficiency: 0.0,
            global_clustering_watts_strogatz: 0.0,
            centralization_closeness: 0.0,
            centralization_triadic_participation: 0.0,
            shortest_path_variance: 0.0,
        }
    }

    /// The main entry point for computing metrics:
    /// 1) Build adjacency list
    /// 2) Compute shortest paths & betweenness
    /// 3) Compute average path length, network efficiency, closeness, clustering, etc.
    pub fn set_network_metrics(&mut self, network2_analyze: &Vec<Vec<bool>>) {
        self.set_shortest_path(&network2_analyze);

        // In Java: double diameter = Double.MIN_VALUE;
        // `Double.MIN_VALUE` in Java is the smallest positive number (~1e-308).
        // Typically for "diameter" we might start at 0.0. We'll replicate the logic by using
        // something very small. Here, we can use f64::MIN_POSITIVE.
        let mut diameter = f64::MIN_POSITIVE;

        self.average_path_length = 0.0;
        self.network_efficiency = 0.0;
        self.centralization_closeness = 0.0;
        self.centralization_triadic_participation = 0.0;
        self.shortest_path_variance = 0.0;
        self.global_clustering_watts_strogatz = 0.0;

        let mut centrality_closeness = vec![0.0; params::N];
        let mut centrality_closeness_max = f64::MIN;
        let mut centrality_triadic = vec![0.0; params::N];
        let mut centrality_triadic_max = f64::MIN;
        let mut shortest_path_sum = vec![0.0; params::N];
        let mut shortest_path_squared_sum = vec![0.0; params::N];

        // Main loop to accumulate statistics
        for i in 0..params::N {
            let degree_i = self.adj_list[i].len();
            for j in i..params::N {
                if i != j {
                    let dist = self.shortest_path[i][j];
                    if dist > 0 {
                        let dist_f = dist as f64;
                        self.average_path_length += dist_f;
                        self.network_efficiency += 1.0 / dist_f;
                        centrality_closeness[i] += dist_f;
                        centrality_closeness[j] += dist_f;
                        shortest_path_sum[i] += dist_f;
                        shortest_path_squared_sum[i] += dist_f * dist_f;

                        if dist_f > diameter {
                            diameter = dist_f;
                        }
                    }
                }
            }
            // After summing distances, compute closeness for node i:
            // closenessCentrality[i] = (N - 1) / closenessCentrality[i]
            // and accumulate for global closeness centralization.
            if centrality_closeness[i] > 0.0 {
                centrality_closeness[i] = (params::N as f64 - 1.0) / centrality_closeness[i];
            }
            self.centralization_closeness -= centrality_closeness[i];
            if centrality_closeness[i] > centrality_closeness_max {
                centrality_closeness_max = centrality_closeness[i];
            }

            // local clustering (Watts-Strogatz)
            if degree_i >= 2 {
                let mut local_clustering_numerator = 0;
                let local_clustering_denominator = degree_i * (degree_i - 1) / 2;

                for &j in &self.adj_list[i] {
                    for &k in &self.adj_list[i] {
                        if j < k && network2_analyze[j][k] {
                            local_clustering_numerator += 1;
                        }
                    }
                }
                self.global_clustering_watts_strogatz +=
                    local_clustering_numerator as f64 / local_clustering_denominator as f64;
                centrality_triadic[i] = local_clustering_numerator as f64;
                self.centralization_triadic_participation -= centrality_triadic[i];
                if centrality_triadic[i] > centrality_triadic_max {
                    centrality_triadic_max = centrality_triadic[i];
                }
            }
        }

        // Compute the variance of shortest paths for each node
        for i in 0..params::N {
            let mean = shortest_path_sum[i] / params::N as f64;
            let mean_square = shortest_path_squared_sum[i] / params::N as f64;
            self.shortest_path_variance += mean_square - (mean * mean);
        }

        // Final normalization
        self.average_path_length /= params::N_DYAD as f64;
        self.network_efficiency /= params::N_DYAD as f64;
        self.centralization_closeness += centrality_closeness_max * (params::N as f64);
        self.centralization_closeness /= params::CLOSENESS_CENTRALIZATION_DENOMINATOR;
        self.centralization_triadic_participation += centrality_triadic_max * (params::N as f64);
        self.centralization_triadic_participation /= params::TRIADIC_CENTRALIZATION_DENOMINATOR;
        self.global_clustering_watts_strogatz /= params::N as f64;
        self.shortest_path_variance /= params::N as f64;
    
        self.adj_list.clear();
    }

    /// Equivalent to `private void setShortestPathAndBetweennessCentrality()`.
    fn set_shortest_path(&mut self, network2_analyze: &Vec<Vec<bool>>) {
        self.shortest_path = vec![vec![-1; params::N]; params::N]; // Use -1 for unvisited
        self.adj_list = vec![Vec::new(); params::N];
        for i in 0..params::N {
            for j in 0..params::N {
                if network2_analyze[i][j] {
                    self.adj_list[i].push(j);
                }
            }
        }
    
        for s in 0..params::N {
            let mut stack: Vec<usize> = Vec::new();
            let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); params::N];
            let mut sigma = vec![0.0; params::N];
            let mut delta = vec![0.0; params::N];
            let mut distance: Vec<isize> = vec![-1; params::N]; // Use -1 for unvisited
    
            sigma[s] = 1.0;
            distance[s] = 0;
    
            let mut queue = VecDeque::new();
            queue.push_back(s);
    
            while let Some(v) = queue.pop_front() {
                stack.push(v);
                for &w in &self.adj_list[v] {
                    if distance[w] == -1 {
                        distance[w] = distance[v] + 1;
                        queue.push_back(w);
                    }
                    if distance[w] == distance[v] + 1 {
                        sigma[w] += sigma[v];
                        predecessors[w].push(v);
                    }
                }
            }
    
            while let Some(w) = stack.pop() {
                for &v in &predecessors[w] {
                    if sigma[w] != 0.0 {
                        delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                    }
                }
            }
    
            for i in 0..params::N {
                self.shortest_path[s][i] = distance[i];
                self.shortest_path[i][s] = distance[i];
            }
        }
    }
    
}
