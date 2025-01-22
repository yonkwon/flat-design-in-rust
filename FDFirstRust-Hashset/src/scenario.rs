use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::cmp;
use std::time::{SystemTime, UNIX_EPOCH};
use crate::params;
use crate::network_analyzer::NetworkAnalyzer;

// --------------------------------------------------------------------
// The Scenario struct in Rust
// --------------------------------------------------------------------
pub struct Scenario {
    // Random generator
    pub rng: rand::rngs::ThreadRng,
    pub tic: usize,

    pub social_dynamics: usize,
    pub is_rewiring: bool,
    pub is_random_rewiring: bool,
    pub is_network_closure: bool,
    pub is_preferential_attachment: bool,

    pub span: usize,            // Span of control
    pub enforcement: f64,       // E

    // New global variables
    pub turbulence_rate: f64,
    pub turnover_rate: f64,

    // Reality replaced by a 1D bool array
    pub reality: Vec<Vec<bool>>,

    pub belief_of: Vec<Vec<Vec<bool>>>,
    pub performance_usize: Vec<usize>,
    pub level_of: Vec<usize>,
    pub level_range: f64,

    // Networks replaced by 2D bool arrays
    pub network: Vec<HashSet<usize>>,
    pub network_formal: Vec<HashSet<usize>>,
    pub network_informal: Vec<HashSet<usize>>,
    pub network_limited: Vec<HashSet<usize>>,
    pub network_analyzer: NetworkAnalyzer,

    pub preference_score: Vec<Vec<f64>>,
    pub preference_score_avg: Vec<f64>,

    pub performance_avg: f64,

    pub average_path_length: f64,
    pub network_efficiency: f64,
    pub global_clustering_watts_strogatz: f64,
    pub overall_centralization: f64,
    pub shortest_path_variance: f64,
    pub sigma: f64,
    pub omega: f64,

    //Utility 
    pub iterator_focal_index: Vec<usize>,
    pub iterator_target_index: Vec<usize>,
    pub iterator_dyad: Vec<(usize, usize)>,

}

impl Scenario {
    pub fn new(
        social_dynamics: usize,
        span: usize,
        enforcement: f64,
        turbulence_rate: f64,
        turnover_rate: f64,
    ) -> Self {
        let tic = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as usize;
        let rng = thread_rng();
        let reality = vec![vec![false; params::M_IN_BUNDLE]; params::M_OF_BUNDLE];
        let belief_of = vec![vec![vec![false; params::M_IN_BUNDLE]; params::M_OF_BUNDLE]; params::N];
        let performance_usize = vec![0; params::N];
        let level_of = vec![0; params::N];
        let network = vec![HashSet::new(); params::N];
        let network_formal = vec![HashSet::new(); params::N];
        let network_informal = vec![HashSet::new(); params::N];
        let network_limited = vec![HashSet::new(); params::N];
        let network_analyzer = NetworkAnalyzer::new();
        let preference_score = vec![vec![0.0; params::N]; params::N];
        let preference_score_avg = vec![0.0; params::N];
        let iterator_focal_index: Vec<usize> = (0..params::N).collect();
        let iterator_target_index = iterator_focal_index.clone();
        let mut iterator_dyad = Vec::with_capacity(params::N_DYAD);

        for i in 0..params::N {
            for j in (i + 1)..params::N {
                iterator_dyad.push((i, j));
            }
        }

        let mut scenario = Scenario{
            tic,
            social_dynamics,
            is_network_closure: false,
            is_preferential_attachment: false,
            is_rewiring: true,
            is_random_rewiring: false,
            span,
            enforcement,
            turbulence_rate,
            turnover_rate,
            rng,
            reality,
            belief_of,
            performance_usize,
            level_of,
            level_range: 0.0,
            network,
            network_formal,
            network_informal,
            network_limited,
            network_analyzer,
            preference_score,
            preference_score_avg,
            performance_avg: 0.0,
            average_path_length: 0.0,
            network_efficiency: 0.0,
            global_clustering_watts_strogatz: 0.0,
            overall_centralization: 0.0,
            shortest_path_variance: 0.0,
            iterator_focal_index,
            iterator_target_index,
            iterator_dyad,
            sigma: 0.0,
            omega: 0.0,
        };

        
        // Set flags based on social_dynamics
        match social_dynamics {
            0 => scenario.is_network_closure = true,
            1 => scenario.is_preferential_attachment = true,
            _ => {}
        }

        // Initialize everything
        scenario.initialize();

        scenario
    }

    /// Equivalent to Java's public Scenario getClone().
    /// Returns a "clone" of the current Scenario with all relevant fields copied.
    pub fn get_clone(&self) -> Scenario {
        // Create the new scenario using the same constructor arguments.
        let mut clone = Scenario::new(
            self.social_dynamics,
            self.span,
            self.enforcement,
            self.turbulence_rate,
            self.turnover_rate,
        );

        clone.reality = self.reality.clone();
        clone.performance_usize = self.performance_usize.clone();
        clone.performance_avg = self.performance_avg;
        clone.belief_of = self.belief_of.clone();
        clone.network_formal = self.network_formal.clone();
        clone.network_informal = self.network_informal.clone();
        clone.network = self.network.clone();
        clone.network_analyzer = NetworkAnalyzer::new();

        clone.set_outcome();

        clone
    }

    /// Equivalent to Java's public Scenario getClone(boolean, boolean).
    pub fn get_clone_with_params(&self, is_rewiring: bool, is_random_rewiring: bool) -> Scenario {
        let mut clone = self.get_clone();
        clone.set_network_params(is_rewiring, is_random_rewiring);
        clone
    }

    /// Equivalent to void setNetworkParams(boolean, boolean).
    pub fn set_network_params(&mut self, is_rewiring: bool, is_random_rewiring: bool) {
        self.is_rewiring = is_rewiring;
        self.is_random_rewiring = is_random_rewiring;
    }

    /// Equivalent to private void initialize().
    fn initialize(&mut self) {
        self.initialize_network();
        self.initialize_entity();
        self.initialize_outcome();
    }

    /// Equivalent to private void initializeNetwork().
    fn initialize_network(&mut self) {
        // Re-initialize them:
        self.network = vec![HashSet::new(); params::N];
        self.network_formal = vec![HashSet::new(); params::N];
        self.network_informal = vec![HashSet::new(); params::N];
        self.network_limited = vec![HashSet::new(); params::N];
        self.network_analyzer = NetworkAnalyzer::new();
        self.level_of = vec![0; params::N];

        let mut level_now = 1;
        let mut upper_start = 0;
        let mut upper_end = 1;
        let mut lower_start = upper_end;
        let mut lower_end = lower_start + self.span as usize;
        self.level_of[0] = 1;

        // Build the hierarchical network
        loop {
            level_now += 1;
            for upper in upper_start..upper_end {
                for lower in lower_start..lower_end {
                    self.network[upper].insert(lower);
                    self.network[lower].insert(upper);
                    self.level_of[lower] = level_now;

                }
                if params::LINK_LEVEL {
                    let lower_num = lower_end - lower_start;
                    if lower_num == 0 {
                        continue;
                    }
                    // Link in a ring among the subordinates
                    for i in 0..lower_num {
                        let focal = lower_start + i;
                        let target = lower_start + ((i + 1) % lower_num);
                        if focal == target {
                            break;
                        }
                        self.network[focal].insert(target);
                        self.network[target].insert(focal);
                    }
                }
                lower_start = lower_end;
                lower_end = cmp::min(lower_start + self.span, params::N);
            }
            if lower_start == params::N {
                break;
            }
            upper_start = upper_end;
            upper_end = upper_start + self.span.pow((level_now - 1) as u32) as usize;
        }
        self.level_range = (level_now - self.level_of[0]) as f64;

        // Tie enforcement
        for focal in 0..params::N {
            for &target in &self.network[focal] {
                if self.rng.gen::<f64>() < self.enforcement {
                    // Enforced
                    self.network_formal[focal].insert(target);
                    self.network_formal[target].insert(focal);
                } else {
                    // Flexible
                    self.network_informal[focal].insert(target);
                    self.network_informal[target].insert(focal);
                }
            }
        }

        // Additional links
        let mut num_addition_left:usize = params::NUM_ADDITION;
        if num_addition_left > 0 {
            self.iterator_dyad.shuffle(&mut thread_rng());
            'outer: loop {
                for &(focal, target) in &self.iterator_dyad {
                    if !self.network[focal].contains(&target)
                        && (self.network_informal[focal].len() < params::INFORMAL_MAX_NUM
                            || self.network_informal[target].len() < params::INFORMAL_MAX_NUM)
                        && num_addition_left > 0
                    {
                        if self.rng.gen::<f64>() < self.enforcement {
                            // Enforced
                            self.network_formal[focal].insert(target);
                            self.network_formal[target].insert(focal);
                        } else {
                            // Flexible
                            self.network_informal[focal].insert(target);
                            self.network_informal[target].insert(focal);
                        }
                        self.network[focal].insert(target);
                        self.network[target].insert(focal);
                        num_addition_left -= 1;
                        if num_addition_left == 0 {
                            break 'outer;
                        }
                    }
                }
                if num_addition_left <= 0 {
                    break 'outer;
                }
            }
        }

        // If Main.params::LIMIT_LEVEL is enabled
        if params::LIMIT_LEVEL {
            for focal in 0..params::N {
                for target in focal..params::N {
                    if (self.level_of[focal] as i32 - self.level_of[target] as i32).abs() > 1 {
                        self.network_limited[focal].insert(focal);
                    }
                }
            }
        }
    }

    /// Equivalent to private void initializeEntity().
    fn initialize_entity(&mut self) {
        // reality and realityBundleID
        for bundle in 0..params::M_OF_BUNDLE {
            for element in 0..params::M_IN_BUNDLE{
                self.reality[bundle][element] = self.rng.gen::<bool>();
                for focal in 0..params::N {
                    self.belief_of[focal][bundle][element] = self.rng.gen::<bool>();
                }
            }
        }
    }

    /// Equivalent to private void initializeOutcome().
    fn initialize_outcome(&mut self) {
        // performance, differenceOf, differenceSum
        for n in 0..params::N {
            self.performance_usize[n] = 0;
        }

        self.set_performance();
        self.set_outcome();
    }

    pub fn step_forward(&mut self){
        if self.is_rewiring{
            if self.is_random_rewiring{
                self.do_random_rewiring(params::INFORMAL_REWIRING_NUM, params::INFORMAL_REWIRING_NUM);
            }else{
                self.do_rewiring(params::INFORMAL_REWIRING_NUM, params::INFORMAL_REWIRING_NUM);
            }
        }
        self.do_learning();
        self.set_outcome();
        if self.turnover_rate > 0.0 {
            self.do_turnover();
        }
    }

    pub fn set_outcome(&mut self) {
        self.performance_avg = 0.0;
        self.network_analyzer.set_network_metrics(&self.network);

        self.average_path_length = self.network_analyzer.get_average_path_length();
        self.network_efficiency = self.network_analyzer.get_network_efficiency();
        self.global_clustering_watts_strogatz = self.network_analyzer.get_global_clustering_watts_strogatz();
        self.overall_centralization = self.network_analyzer.get_global_closeness_centralization();
        self.shortest_path_variance = self.network_analyzer.get_shortest_path_variance();

        for focal in 0..params::N {
            self.performance_avg += self.performance_usize[focal] as f64;
        }
        self.performance_avg /= params::M_N as f64;
    }

    /// Equivalent to double getRewiringWeight(int focal, int target).
    fn get_rewiring_weight(&self, focal: usize, target: usize) -> f64 {
        if self.is_network_closure {
            self.get_rewiring_weight_network_closure(focal, target)
        } else if self.is_preferential_attachment {
            self.get_rewiring_weight_preferential_attachment(focal, target)
        } else {
            0.0
        }
    }

    /// Equivalent to double getRewiringWeightNetworkClosure(int focal, int target).
    fn get_rewiring_weight_network_closure(&self, focal: usize, target: usize) -> f64 {
        let mut preference_score = 1.0; // to avoid weight of 0
        for i in 0..params::N {
            if self.network[focal].contains(&i) && self.network[target].contains(&i) {
                preference_score += 1.0;
            }
        }
        let denom = cmp::max(self.network[focal].len(), self.network[target].len()) as f64
            + if self.network[focal].contains(&target) { 0.0 } else { 1.0 };
        preference_score / denom
    }

    fn get_rewiring_weight_preferential_attachment(&self, focal: usize, target: usize) -> f64 {
        let df = self.network[focal].len() as f64;
        let dt = self.network[target].len() as f64;
        df.min(dt)
    }

    pub fn do_rewiring(&mut self, num_formation: usize, num_break: usize) {
        if self.is_random_rewiring {
            self.do_random_rewiring(num_formation, num_break);
        } else {
            self.do_tie_formation(num_formation);
            self.do_tie_break(num_break);
        }
    }

    fn do_tie_break(&mut self, mut num_break: usize) {
        while num_break > 0 {
            let mut all_pairs_informal: Vec<(usize, usize)> = self.network_informal
                .iter()
                .enumerate()
                .flat_map(|(focal, targets)| {
                    targets.iter().map(move |&target| (focal, target))
                })
                .collect();
            all_pairs_informal.shuffle(&mut self.rng);
    
            let mut probability = vec![0.0; all_pairs_informal.len()];
            let mut dyad2_cut_weight_max = f64::MIN;
            // Calculate rewiring weights
            for (id, &(focal, target)) in all_pairs_informal.iter().enumerate() {
                let w = self.get_rewiring_weight(focal, target);
                probability[id] = w;
                if w > dyad2_cut_weight_max {
                    dyad2_cut_weight_max = w;
                }
            }
            // Convert to "largest becomes zero" style
            let mut probability_denominator = 0.0;
            for prob in &mut probability {
                if *prob != 0.0 {
                    *prob = dyad2_cut_weight_max - *prob;
                    probability_denominator += *prob;
                }
            }
            if probability_denominator == 0.0 {
                probability.fill(1.0);
                probability_denominator = probability.len() as f64;
            }
            // Choose
            let marker = self.rng.gen::<f64>();
            let mut probability_cum = 0.0;
            for (id, &(focal, target)) in all_pairs_informal.iter().enumerate() {
                if probability[id] != 0.0 {
                    probability_cum += probability[id] / probability_denominator;
                    if probability_cum >= marker {
                        self.network[focal].remove(&target);
                        self.network[target].remove(&focal);
                        self.network_informal[focal].remove(&target);
                        self.network_informal[target].remove(&focal);
                        num_break -= 1;
                        break;
                    }
                }
            }
        }
    }

    fn do_tie_formation(&mut self, mut num_formation: usize) {
        while num_formation > 0 {
            let mut probability = vec![0.0; params::N_DYAD];
            let mut probability_denominator = 0.0;
    
            // Collect probabilities
            for (d, (focal, target)) in self.iterator_dyad.iter().enumerate() {
                if !self.network[*focal].contains(target)
                    && focal != target
                    && !self.network_limited[*focal].contains(target)
                    && self.network_informal[*focal].len() < params::INFORMAL_MAX_NUM
                    && self.network_informal[*target].len() < params::INFORMAL_MAX_NUM
                {
                    let w = self.get_rewiring_weight(*focal, *target);
                    probability[d] = w;
                    probability_denominator += w;
                }
            }
    
            if probability_denominator == 0.0 {
                probability.fill(1.0);
                probability_denominator = self.iterator_dyad.len() as f64;
            }
    
            let marker = self.rng.gen::<f64>();
            let mut probability_cum = 0.0;
            for (d, (focal, target)) in self.iterator_dyad.iter().enumerate() {
                if probability[d] != 0.0 {
                    probability_cum += probability[d] / probability_denominator;
                    if probability_cum >= marker {
                        self.network[*focal].insert(*target);
                        self.network[*target].insert(*focal);
                        self.network_informal[*focal].insert(*target);
                        self.network_informal[*target].insert(*focal);
                        num_formation -= 1;
                        break;
                    }
                }
            }
        }
    }

    fn do_random_rewiring(&mut self, mut num_formation: usize, mut num_break: usize) {
        while num_formation > 0 || num_break > 0 {
            self.iterator_dyad.shuffle(&mut self.rng);
            for (focal, target) in &self.iterator_dyad {
                if self.network_informal[*focal].contains(target) && num_break > 0 {
                    // Remove this informal tie
                    self.network[*focal].remove(target);
                    self.network[*target].remove(focal);
                    self.network_informal[*focal].remove(target);
                    self.network_informal[*target].remove(focal);
                    num_break -= 1;
                } else if num_formation > 0
                    && !self.network[*focal].contains(target)
                    && focal != target
                    && (self.network_informal[*focal].len() < params::INFORMAL_MAX_NUM
                        || self.network_informal[*target].len() < params::INFORMAL_MAX_NUM)
                    && !self.network_limited[*focal].contains(target)
                {
                    // Form this informal tie
                    self.network[*focal].insert(*target);
                    self.network[*target].insert(*focal);
                    self.network_informal[*focal].insert(*target);
                    self.network_informal[*target].insert(*focal);
                    num_formation -= 1;
                }
                if num_formation == 0 && num_break == 0 {
                    break;
                }
            }
            if num_formation > 0 || num_break > 0 {
                println!("\tRandom Rewiring Reiterated");
            }
        }
    }

    fn do_learning(&mut self) {
        let mut belief_changes = Vec::new();
        let mut majority_opinion_count = vec![0; params::M_IN_BUNDLE];
        for focal in 0..params::N {
            for bundle in 0..params::M_OF_BUNDLE {
                majority_opinion_count.fill(0);
                // Tally majority from better-performing neighbors
                for &target in &self.network[focal] {
                    if self.performance_usize[target] > self.performance_usize[focal] {
                        for element in 0..params::M_IN_BUNDLE {
                            if self.belief_of[target][bundle][element] {
                                majority_opinion_count[element] += 1;
                            } else {
                                majority_opinion_count[element] -= 1;
                            }
                        }
                    }
                }
                // Decide which bits to flip
                for element in 0..params::M_IN_BUNDLE {
                    if majority_opinion_count[element] > 0 && !self.belief_of[focal][bundle][element] && self.rng.gen::<f64>() < params::P_LEARNING {
                        belief_changes.push((focal, bundle, element, true));
                    } else if majority_opinion_count[element] < 0 && self.belief_of[focal][bundle][element] && self.rng.gen::<f64>() < params::P_LEARNING {
                        belief_changes.push((focal, bundle, element, false));
                    }
                }
            }
        }
        // Apply changes
        for &(focal, bundle, element, new_belief) in &belief_changes {
            self.belief_of[focal][bundle][element] = new_belief;
        }
        // Update performance for all focal nodes
        self.set_performance();
    }

    /// Equivalent to int getPerformance(int focal).
    fn get_performance(&self, focal: usize) -> usize {
        let belief_of_focal = &self.belief_of[focal];
        self.reality.iter()
            .enumerate()
            .filter(|(bundle_idx, bundle)| {
                bundle.iter().zip(&belief_of_focal[*bundle_idx]).all(|(r, b)| r == b)
            })
            .count() * params::M_IN_BUNDLE
    }

    /// Equivalent to void setPerformance(int focal).
    fn set_performance_of(&mut self, focal: usize) {
        self.performance_usize[focal] = self.get_performance(focal);
    }

    /// Equivalent to void setPerformance().
    fn set_performance(&mut self) {
        for focal in 0..params::N {
            self.set_performance_of(focal);
        }
    }

    /// Equivalent to void printCSV(String fileName).
    pub fn export_network_csv(&self, file_name: &str) {
        let out_file = match File::create(format!("{}.csv", file_name)) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Failed to create file: {}", e);
                return;
            }
        };
        let mut writer = BufWriter::new(out_file);

        // Header
        writeln!(&mut writer, "SOURCE,TARGET,TIE_ENFORCED").unwrap();

        // Edges
        for focal in 0..params::N {
            for target in focal..params::N {
                if focal == target {
                    continue;
                }
                if self.network[focal].contains(&target) {
                    let tie_enforced = self.network_formal[focal].contains(&target);
                    writeln!(
                        &mut writer,
                        "{},{},{}",
                        focal, target, tie_enforced
                    )
                    .unwrap();
                }
            }
        }

        // Individual lines
        for focal in 0..params::N {
            writeln!(&mut writer, "{},,", focal).unwrap();
        }
    }

    pub fn do_turnover(&mut self) {
        for n in 0..params::N {
            if self.rng.gen::<f64>() < self.turnover_rate {
                // The individual n leaves; fill with new random beliefs
                for bundle in 0..params::M_OF_BUNDLE {
                    for element in 0..params::M_IN_BUNDLE {
                        self.belief_of[n][bundle][element] = self.rng.gen::<bool>();
                    }
                }
                // Recompute performance for this individual
                self.set_performance_of(n);
            }
        }
    }
    
    /// doTurbulence(): each dimension of reality is flipped with probability turbulenceRate.
    pub fn do_turbulence(&mut self) {
        for bundle in 0..params::M_OF_BUNDLE {
            for element in 0..params::M_IN_BUNDLE {
                if self.rng.gen::<f64>() < self.turbulence_rate {
                    self.reality[bundle][element] = !self.reality[bundle][element];
                }
            }
        }
        self.set_performance();
    }
}