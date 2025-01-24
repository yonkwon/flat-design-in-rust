use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::{cmp, usize};
use std::time::{SystemTime, UNIX_EPOCH};
use crate::params;
use crate::network_analyzer::{self, NetworkAnalyzer};

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
    pub performance_of: Vec<usize>,
    pub level_of: Vec<usize>,
    pub level_range: f64,

    // Networks replaced by 2D bool arrays
    pub network: Vec<Vec<bool>>,
    pub network_formal: Vec<Vec<bool>>,
    pub network_informal: Vec<Vec<bool>>,
    pub network_limited: Vec<Vec<bool>>,
    pub network_analyzer: crate::network_analyzer::NetworkAnalyzer,

    pub degree: Vec<isize>,
    pub degree_formal: Vec<isize>,
    pub degree_informal: Vec<isize>,

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
        let tic =  SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as usize;
        let rng = thread_rng();
        let reality = vec![vec![false; params::M_IN_BUNDLE]; params::M_OF_BUNDLE];
        let belief_of = vec![vec![vec![false; params::M_IN_BUNDLE]; params::M_OF_BUNDLE]; params::N];
        let performance_usize = vec![0; params::N];
        let level_of = vec![0; params::N];
        let network = vec![vec![false; params::N]; params::N];
        let network_formal = network.clone();
        let network_informal = network.clone();
        let network_limited = network.clone();
        let network_analyzer = NetworkAnalyzer::new();
        let degree = vec![0;params::N];
        let degree_formal = degree.clone();
        let degree_informal = degree.clone();
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
            rng,
            tic,
            social_dynamics,
            is_rewiring: true,
            is_random_rewiring: false,
            is_network_closure: false,
            is_preferential_attachment: false,
            span,
            enforcement,
            turbulence_rate,
            turnover_rate,
            reality,
            belief_of,
            performance_of: performance_usize,
            level_of,
            level_range: 0.0,
            network,
            network_formal,
            network_informal,
            network_limited,
            network_analyzer,
            degree,
            degree_formal,
            degree_informal,
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
        clone.performance_of = self.performance_of.clone();
        clone.performance_avg = self.performance_avg;
        clone.belief_of = self.belief_of.clone();
        clone.network_formal = self.network_formal.clone();
        clone.network_informal = self.network_informal.clone();
        clone.network = self.network.clone();
        clone.network_analyzer = NetworkAnalyzer::new();
        clone.degree = self.degree.clone();
        clone.degree_formal = self.degree_formal.clone();
        clone.degree_informal = self.degree_informal.clone();

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
        self.network = vec![vec![false; params::N]; params::N];
        self.network_formal = vec![vec![false; params::N]; params::N];
        self.network_informal = vec![vec![false; params::N]; params::N];
        self.network_limited = vec![vec![false; params::N]; params::N];
        self.level_of = vec![0; params::N];
        self.degree = vec![0; params::N];
        self.degree_formal = vec![0; params::N];
        self.degree_informal = vec![0; params::N];

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
                    self.network[upper][lower] = true;
                    self.network[lower][upper] = true;
                    self.degree[upper] += 1;
                    self.degree[lower] += 1;
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
                        self.network[focal][target] = true;
                        self.network[target][focal] = true;
                        self.degree[focal] += 1;
                        self.degree[target] += 1;
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
            for target in focal..params::N {
                if self.network[focal][target] {
                    if self.rng.gen::<f64>() < self.enforcement {
                        // Enforced
                        self.network_formal[focal][target] = true;
                        self.network_formal[target][focal] = true;
                        self.degree_formal[focal] += 1;
                        self.degree_formal[target] += 1;
                    } else {
                        // Flexible
                        self.network_informal[focal][target] = true;
                        self.network_informal[target][focal] = true;
                        self.degree_informal[focal] += 1;
                        self.degree_informal[target] += 1;
                    }
                }
            }
        }

        // Additional links
        let mut num_addition_left:usize = params::NUM_ADDITION;
        if num_addition_left > 0 {
            self.iterator_dyad.shuffle(&mut thread_rng());
            'outer: loop {
                for &(focal, target) in &self.iterator_dyad {
                    if !self.network[focal][target]
                        && (self.degree_informal[focal] < params::INFORMAL_MAX_NUM
                            || self.degree_informal[target] < params::INFORMAL_MAX_NUM)
                        && num_addition_left > 0
                    {
                        self.network[focal][target] = true;
                        self.network[target][focal] = true;
                        if self.rng.gen::<f64>() < self.enforcement {
                            // Enforced
                            self.network_formal[focal][target] = true;
                            self.network_formal[target][focal] = true;
                        } else {
                            // Flexible
                            self.network_informal[focal][target] = true;
                            self.network_informal[target][focal] = true;
                        }
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

        if params::LIMIT_LEVEL {
            for &focal in self.iterator_focal_index.iter() {
                self.network_limited[focal][focal] = true;
                for target in focal..params::N {
                    if (self.level_of[focal] as i32 - self.level_of[target] as i32).abs() > 1 {
                        self.network_limited[focal][focal] = true;
                    }
                }
            }
        }

        // println!("\n\ns{} {} <- {}", self.span, self.network.iter().flatten().map(|&x| x as usize).sum::<usize>(), format!("{:?}",self.network));
        // println!("\n\nINFORMAL\ts{} {} <- {}", self.span, self.network_informal.iter().flatten().map(|&x| x as usize).sum::<usize>(), format!("{:?}",self.network_informal));

        self.network_analyzer = network_analyzer::NetworkAnalyzer::new();

        
    }

    fn initialize_entity(&mut self) {
        for bundle in 0..params::M_OF_BUNDLE {
            for element in 0..params::M_IN_BUNDLE{
                self.reality[bundle][element] = self.rng.gen::<bool>();
                for focal in 0..params::N {
                    self.belief_of[focal][bundle][element] = self.rng.gen::<bool>();
                }
            }
        }
    }

    fn initialize_outcome(&mut self) {
        for n in 0..params::N {
            self.performance_of[n] = 0;
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
        self.network_analyzer.set_network_metrics();
        self.average_path_length = self.network_analyzer.get_average_path_length();
        self.network_efficiency = self.network_analyzer.get_network_efficiency();
        self.global_clustering_watts_strogatz = self.network_analyzer.get_global_clustering_watts_strogatz();
        self.overall_centralization = self.network_analyzer.get_global_closeness_centralization();
        self.shortest_path_variance = self.network_analyzer.get_shortest_path_variance();
        for focal in 0..params::N {
            self.performance_avg += self.performance_of[focal] as f64;
        }
        self.performance_avg /= params::M_N as f64;
    }

    pub fn set_preference_score(&mut self) {
        self.preference_score = vec![vec![0.0; params::N]; params::N];
        self.preference_score_avg = vec![0.0; params::N];

        if self.is_network_closure {
            self.set_preference_score_raw_network_closure();
        } else if self.is_preferential_attachment {
            self.set_preference_score_raw_preferential_attachment();
        }
    }

    fn set_preference_score_raw_network_closure(&mut self) {
        for &focal in self.iterator_focal_index.iter() {
            for target in focal..params::N {
                let mut score = 0.0;
                for i in 0..params::N {
                    if self.network[i][focal] && self.network[i][target] {
                        score += 1.0;
                    }
                }
                self.preference_score[focal][target] = score;
                self.preference_score[target][focal] = score;
            }
        }
        for &focal in self.iterator_focal_index.iter() {
            self.preference_score[focal][focal] = 0.0;
            for &target in self.iterator_target_index.iter() {
                let denom = if self.network[focal][target] {
                    self.degree[focal]
                } else {
                    self.degree[focal] + 1
                } as f64;
                self.preference_score[focal][target] /= denom;
                self.preference_score_avg[focal] += self.preference_score[focal][target];
            }
            // Avoid dividing by zero if degree == 0
            if self.degree[focal] > 0 {
                self.preference_score_avg[focal] /= self.degree[focal] as f64;
            }
        }
    }

    fn set_preference_score_raw_preferential_attachment(&mut self) {
        for &focal in self.iterator_focal_index.iter() {
            for target in focal..params::N {
                let d_target = self.degree[target] as f64;
                let d_focal = self.degree[focal] as f64;
                self.preference_score[focal][target] = d_target;
                self.preference_score[target][focal] = d_focal;
            }
        }
        for &focal in self.iterator_focal_index.iter() {
            self.preference_score[focal][focal] = 0.0;
            for &target in self.iterator_target_index.iter() {
                self.preference_score_avg[focal] += self.preference_score[focal][target];
            }
            if self.degree[focal] > 0 {
                self.preference_score_avg[focal] /= self.degree[focal] as f64;
            }
        }
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
            if self.network[focal][i] && self.network[target][i] {
                preference_score += 1.0;
            }
        }
        let denom = cmp::max(self.degree[focal], self.degree[target]) as f64
            + if self.network[focal][target] { 0.0 } else { 1.0 };
        preference_score / denom
    }

    fn get_rewiring_weight_preferential_attachment(&self, focal: usize, target: usize) -> f64 {
        let df = self.degree[focal] as f64;
        let dt = self.degree[target] as f64;
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
        let mut probability = vec![0.0; params::N_DYAD];
        while num_break > 0 {
            probability.fill(0.0);
            let mut dyad2_cut_weight_max = f64::MIN;
            // Calculate rewiring weights
            for (d, (focal, target)) in self.iterator_dyad.iter().enumerate() {
                if self.network_informal[*focal][*target] {
                    let w = self.get_rewiring_weight(*focal, *target);
                    probability[d] = w;
                    if w > dyad2_cut_weight_max {
                        dyad2_cut_weight_max = w;
                    }
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
                probability_denominator = 1.0;
                self.iterator_dyad.shuffle(&mut self.rng);
            }
            // Choose
            let marker = self.rng.gen::<f64>();
            let mut probability_cum = 0.0;
            for (d, (focal, target)) in self.iterator_dyad.iter().enumerate() {
                if probability[d] != 0.0 {
                    probability_cum += probability[d] / probability_denominator;
                    if probability_cum >= marker {
                        self.network[*focal][*target] = false;
                        self.network[*target][*focal] = false;
                        self.network_informal[*focal][*target] = false;
                        self.network_informal[*target][*focal] = false;
                        self.degree[*focal] -= 1;
                        self.degree_informal[*focal] -= 1;
                        self.degree[*target] -= 1;
                        self.degree_informal[*target] -= 1;
                        num_break -= 1;
                        break;
                    }
                }
            }
        }
    }

    /// Equivalent to void doTieFormation(int numFormation).
    fn do_tie_formation(&mut self, mut num_formation: usize) {
        while num_formation > 0 {
            let mut probability = vec![0.0; params::N_DYAD];
            let mut probability_denominator = 0.0;
            // Collect probabilities
            for (d, (focal, target)) in self.iterator_dyad.iter().enumerate() {
                if !self.network[*focal][*target]
                    && *focal != *target
                    && !self.network_limited[*focal][*target]
                    && self.degree_informal[*focal] < params::INFORMAL_MAX_NUM
                    && self.degree_informal[*target] < params::INFORMAL_MAX_NUM
                {
                    probability[d] = self.get_rewiring_weight(*focal, *target);
                    probability_denominator += probability[d];
                }
            }
            if probability_denominator == 0.0 {
                probability.fill(1.0);
                probability_denominator = 1.0;
                self.iterator_dyad.shuffle(&mut self.rng);
            }
            let marker = self.rng.gen::<f64>();
            let mut probability_cum = 0.0;
            for (d, (focal, target)) in self.iterator_dyad.iter().enumerate() {
                if probability[d] != 0.0 {
                    probability_cum += probability[d] / probability_denominator;
                    if probability_cum >= marker {
                        self.network[*focal][*target] = true;
                        self.network[*target][*focal] = true;
                        self.network_informal[*focal][*target] = true;
                        self.network_informal[*target][*focal] = true;
                        self.degree[*focal] += 1;
                        self.degree_informal[*focal] += 1;
                        self.degree[*target] += 1;
                        self.degree_informal[*target] += 1;
                        num_formation -= 1;
                        break;
                    }
                }
            }
        }
    }

    /// Equivalent to void doRandomRewiring(int numFormation, int numBreak).
    fn do_random_rewiring(&mut self, mut num_formation: usize, mut num_break: usize) {
        let mut keep_going = true;
        while keep_going {
            self.iterator_dyad.shuffle(&mut self.rng);
            for (focal, target) in &self.iterator_dyad {
                if self.network_informal[*focal][*target] && num_break > 0 {
                    // Remove this informal tie
                    self.network[*focal][*target] = false;
                    self.network[*target][*focal] = false;
                    self.network_informal[*focal][*target] = false;
                    self.network_informal[*target][*focal] = false;
                    self.degree[*focal] -= 1;
                    self.degree_informal[*focal] -= 1;
                    self.degree[*target] -= 1;
                    self.degree_informal[*target] -= 1;
                    num_break -= 1;
                } else if num_formation > 0
                    && !self.network[*focal][*target]
                    && focal != target
                    && (self.degree_informal[*focal] < params::INFORMAL_MAX_NUM
                        || self.degree_informal[*target] < params::INFORMAL_MAX_NUM)
                    && !self.network_limited[*focal][*target]
                {
                    self.network[*focal][*target] = true;
                    self.network[*target][*focal] = true;
                    self.network_informal[*focal][*target] = true;
                    self.network_informal[*target][*focal] = true;
                    self.degree[*focal] += 1;
                    self.degree_informal[*focal] += 1;
                    self.degree[*target] += 1;
                    self.degree_informal[*target] += 1;
                    num_formation -= 1;
                }
                if num_formation == 0 && num_break == 0 {
                    break;
                }
            }
            if num_formation == 0 && num_break == 0 {
                keep_going = false;
            } else {
                println!("\tRandom Rewiring Reiterated");
            }
        }
    }

    fn do_learning(&mut self) {
        let mut majority_opinion_count = vec![vec![vec![0; params::M_IN_BUNDLE]; params::M_OF_BUNDLE]; params::N];
        for (focal, target) in &self.iterator_dyad {
            let (superior, inferior) = if self.network[*focal][*target] && self.performance_of[*focal] != self.performance_of[*target] {
                if self.performance_of[*focal] > self.performance_of[*target] {
                    (*focal, *target)
                } else{
                    (*target, *focal)
                }
            } else {
                continue;
            };
            for bundle in 0..params::M_OF_BUNDLE{
                for element in 0..params::M_IN_BUNDLE{
                    majority_opinion_count[inferior][bundle][element] += if self.belief_of[superior][bundle][element] { 1 } else { -1 };
                }
            }
        }
        for focal in 0..params::N {
            for bundle in 0..params::M_OF_BUNDLE {
                let beliefs = &mut self.belief_of[focal][bundle];
                let counts = &majority_opinion_count[focal][bundle];
                for element in 0..params::M_IN_BUNDLE {
                    let count = counts[element];
                    let belief = beliefs[element];
                    if (count > 0 && belief) || (count < 0 && !belief) || count == 0 {
                        continue;
                    }
                    if self.rng.gen::<f64>() < params::P_LEARNING {
                        beliefs[element] = !belief;
                    }
                }
            }
            self.set_performance_of(focal);
        }
    }

    fn get_performance_of(&self, focal: usize) -> usize {
        let belief_of_focal = &self.belief_of[focal];
        let mut performance_of_focal:usize = 0;
        'bundle: for bundle in 0..params::M_OF_BUNDLE{
            for element in 0..params::M_IN_BUNDLE{
                if self.reality[bundle][element] != belief_of_focal[bundle][element] {
                    continue 'bundle;
                }
                performance_of_focal += params::M_IN_BUNDLE;
            }
        }
        performance_of_focal
    }

    /// Equivalent to void setPerformance(int focal).
    fn set_performance_of(&mut self, focal: usize) {
        self.performance_of[focal] = self.get_performance_of(focal);
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
                if self.network[focal][target] {
                    let tie_enforced = self.network_formal[focal][target];
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

    /// doTurnover(): each individual leaves with probability turnoverRate; replaced by a new random one.
    pub fn do_turnover(&mut self) {
        for focal in 0..params::N {
            if self.rng.gen::<f64>() < self.turnover_rate {
                for bundle in 0..params::M_OF_BUNDLE {
                    for element in 0..params::M_IN_BUNDLE{
                        self.belief_of[focal][bundle][element] = self.rng.gen::<bool>();
                    }
                }
                self.set_performance_of(focal);
            }
        }
    }

    /// doTurbulence(): each dimension of reality is flipped with probability turbulenceRate.
    pub fn do_turbulence(&mut self) {
        for bundle in 0..params::M_OF_BUNDLE {
            for element in 0..params::M_IN_BUNDLE{
                if self.rng.gen::<f64>() < self.turbulence_rate {
                    self.reality[bundle][element] = !self.reality[bundle][element];
                }
            }
        }
        self.set_performance();
    }
}