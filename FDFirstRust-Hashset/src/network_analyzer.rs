    use crate::params;
    use std::collections::VecDeque;
    use std::collections::HashSet;
    use std::{f64, usize};

    pub struct NetworkAnalyzer {
        pub shortest_path: Vec<Vec<isize>>, // declared as an signed integer to allow for -1 as a procedural marker.
        pub average_path_length: f64,
        pub network_efficiency: f64,
        pub global_clustering_watts_strogatz: f64,
        pub global_closeness_centralization: f64,
        pub shortest_path_variance: f64,
    }

    impl NetworkAnalyzer {

        pub fn new() -> Self {
            NetworkAnalyzer {
                shortest_path: vec![vec![-1; params::N]; params::N],
                average_path_length: 0.0,
                network_efficiency: 0.0,
                global_clustering_watts_strogatz: 0.0,
                global_closeness_centralization: 0.0,
                shortest_path_variance: 0.0,
            }
        }

        pub fn set_network_metrics(&mut self, network2_analyze: &Vec<HashSet<usize>>) {
            self.set_shortest_path(network2_analyze);

            self.average_path_length = 0.0;
            self.network_efficiency = 0.0;
            self.global_closeness_centralization = 0.0;
            self.shortest_path_variance = 0.0;
            self.global_clustering_watts_strogatz = 0.0;

            let mut closeness_centrality_max = f64::MIN;
            let mut closeness_centrality = vec![0.0; params::N];
            let mut shortest_path_sum = vec![0.0; params::N];
            let mut shortest_path_squared_sum = vec![0.0; params::N];

            // Main loop to accumulate statistics
            for i in 0..params::N {
                let degree_i = network2_analyze[i].len();
                for j in (i + 1)..params::N {
                    if i != j {
                        let dist = self.shortest_path[i][j];
                        if dist > 0 {
                            let dist_f = dist as f64;
                            self.average_path_length += dist_f;
                            self.network_efficiency += 1.0 / dist_f;
                            closeness_centrality[i] += dist_f;
                            closeness_centrality[j] += dist_f;
                            shortest_path_sum[i] += dist_f;
                            shortest_path_squared_sum[i] += dist_f * dist_f;
                        }
                    }
                }
                // After summing distances, compute closeness for node i:
                if closeness_centrality[i] > 0.0 {
                    closeness_centrality[i] = (params::N as f64 - 1.0) / closeness_centrality[i];
                }
                self.global_closeness_centralization -= closeness_centrality[i];
                if closeness_centrality[i] > closeness_centrality_max {
                    closeness_centrality_max = closeness_centrality[i];
                }
                if degree_i >= 2 {
                    let mut local_clustering_numerator = 0;
                    let local_clustering_denominator = degree_i * (degree_i - 1) / 2;

                    for &j in &network2_analyze[i] {
                        for &k in &network2_analyze[i] {
                            if j < k && network2_analyze[j].contains(&k) {
                                local_clustering_numerator += 1;
                            }
                        }
                    }
                    self.global_clustering_watts_strogatz +=
                        local_clustering_numerator as f64 / local_clustering_denominator as f64;
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
            self.global_closeness_centralization += closeness_centrality_max * (params::N as f64);
            self.global_closeness_centralization /= params::CLOSENESS_CENTRALIZATION_DENOMINATOR;
            self.global_clustering_watts_strogatz /= params::N as f64;
            self.shortest_path_variance /= params::N as f64;
        }

        fn set_shortest_path(&mut self, network2_analyze: &Vec<HashSet<usize>>) {
            self.shortest_path = vec![vec![-1; params::N]; params::N]; // Use -1 for unvisited
        
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
                    for &w in &network2_analyze[v] {
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

        pub fn get_average_path_length(&self) -> f64 {
            self.average_path_length
        }

        pub fn get_network_efficiency(&self) -> f64 {
            self.network_efficiency
        }

        pub fn get_global_clustering_watts_strogatz(&self) -> f64 {
            self.global_clustering_watts_strogatz
        }

        pub fn get_global_closeness_centralization(&self) -> f64 {
            self.global_closeness_centralization
        }

        pub fn get_shortest_path_variance(&self) -> f64 {
            self.shortest_path_variance
        }

    }