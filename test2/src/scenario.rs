use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::cmp;
use crate::params;
use crate::network_analyzer;

// --------------------------------------------------------------------
// The Scenario struct in Rust
// --------------------------------------------------------------------
pub struct Scenario {
    // Random generator
    pub rng: rand::rngs::ThreadRng,

    pub social_dynamics: usize,
    pub is_rewiring: bool,
    pub is_random_rewiring: bool,
    pub is_network_closure: bool,
    pub is_preferential_attachment: bool,

    pub span: usize,              // Span of control
    pub enforcement: f64,       // E

    // New global variables
    pub turbulence_rate: f64,
    pub turnover_rate: f64,

    // Reality replaced by a 1D bool array
    pub reality: Vec<bool>,
    pub reality_bundle_id: Vec<usize>,

    pub belief_of: Vec<Vec<bool>>,
    pub performance_usize: Vec<usize>,
    pub level_of: Vec<usize>,
    pub level_range: f64,

    // Networks replaced by 2D bool arrays
    pub network: Vec<Vec<bool>>,
    pub network_formal: Vec<Vec<bool>>,
    pub network_informal: Vec<Vec<bool>>,
    pub network_limited: Vec<Vec<bool>>,
    pub na: crate::network_analyzer::NetworkAnalyzer,

    pub degree: Vec<usize>,
    pub degree_formal: Vec<usize>,
    pub degree_informal: Vec<usize>,

    pub preference_score: Vec<Vec<f64>>,
    pub preference_score_avg: Vec<f64>,

    pub performance_avg: f64,

    pub average_path_length: f64,
    pub network_efficiency: f64,
    pub global_clustering_watts_strogatz: f64,
    pub overall_centralization: f64,
    pub shortest_path_variance: f64,

    //Utility 
    pub iterator_focal_index: Vec<usize>,
    pub iterator_target_index: Vec<usize>,
    pub iterator_dyad_index: Vec<usize>,
    pub map_dyad2d_index: Vec<[usize; 2]>
}

impl Scenario {
    pub fn new(
        social_dynamics: usize,
        span: usize,
        enforcement: f64,
        turbulence_rate: f64,
        turnover_rate: f64,
    ) -> Self {
        let mut scenario = Scenario{
            social_dynamics,
            is_network_closure: false,
            is_preferential_attachment: false,
            is_rewiring: true,
            is_random_rewiring: false,
            span,
            enforcement,
            turbulence_rate,
            turnover_rate,
            rng: thread_rng(),
            na: crate::network_analyzer::NetworkAnalyzer::new(),
            reality: vec![false; params::N],
            reality_bundle_id: vec![0; params::N],
            belief_of: vec![vec![false; params::N]; params::N],
            performance_usize: vec![0; params::N],
            level_of: vec![0; params::N],
            level_range: 0.0,
            network: vec![vec![false; params::N]; params::N],
            network_formal: vec![vec![false; params::N]; params::N],
            network_informal: vec![vec![false; params::N]; params::N],
            network_limited: vec![vec![false; params::N]; params::N],
            degree: vec![0; params::N],
            degree_formal: vec![0; params::N],
            degree_informal: vec![0; params::N],
            preference_score: vec![vec![0.0; params::N]; params::N],
            preference_score_avg: vec![0.0; params::N],
            performance_avg: 0.0,
            average_path_length: 0.0,
            network_efficiency: 0.0,
            global_clustering_watts_strogatz: 0.0,
            overall_centralization: 0.0,
            shortest_path_variance: 0.0,
            iterator_focal_index: vec![0; params::N].into_iter().collect(),
            iterator_target_index: vec![0; params::N].into_iter().collect(),
            iterator_dyad_index: vec![0; params::N_DYAD].into_iter().collect(),
            map_dyad2d_index:vec![[0; 2]; params::N_DYAD].into_iter().collect(),
        };
        
        
        for i in 0..params::N {
            scenario.iterator_focal_index[i] = i;
        }
        scenario.iterator_target_index = scenario.iterator_focal_index.clone();

        scenario.iterator_dyad_index = vec![0; params::N_DYAD];
        scenario.map_dyad2d_index = vec![[0; 2]; params::N_DYAD];

        let mut d = 0;
        for i in 0..params::N {
            for j in i..params::N {
                if i == j {
                    continue;
                }
                scenario.iterator_dyad_index[d] = d;
                scenario.map_dyad2d_index[d][0] = i;
                scenario.map_dyad2d_index[d][1] = j;
                d += 1;
            }
        }

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
        clone.reality_bundle_id = self.reality_bundle_id.clone();
        clone.performance_usize = self.performance_usize.clone();
        clone.performance_avg = self.performance_avg;

        // Clone beliefOf
        for focal in 0..params::N {
            clone.belief_of[focal] = self.belief_of[focal].clone();

            clone.network_formal[focal] = self.network_formal[focal].clone();
            clone.network_informal[focal] = self.network_informal[focal].clone();
            clone.network[focal] = self.network[focal].clone();
        }

        // Clone degrees
        clone.degree = self.degree.clone();
        clone.degree_formal = self.degree_formal.clone();
        clone.degree_informal = self.degree_informal.clone();

        // Re-initialize the network analyzer with the clone's network
        clone.na = network_analyzer::NetworkAnalyzer::new();

        /////////////HOW CAN IT LINK THE NETWORK TO NETOWRKANALYZER

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
        // network, networkFormal, networkInformal, networkLimited all exist as 2D bool arrays

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
            self.iterator_dyad_index.shuffle(&mut thread_rng());
            'outer: loop {
                for &dyad in self.iterator_dyad_index.iter() {
                    let focal = self.map_dyad2d_index[dyad][0];
                    let target = self.map_dyad2d_index[dyad][1];
                    if !self.network[focal][target]
                        && (self.degree_informal[focal] < params::INFORMAL_MAX_NUM
                            || self.degree_informal[target] < params::INFORMAL_MAX_NUM)
                        && num_addition_left > 0
                    {
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
                        self.network[focal][target] = true;
                        self.network[target][focal] = true;
                        self.degree[focal] += 1;
                        self.degree[target] += 1;
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
            for &focal in self.iterator_focal_index.iter() {
                self.network_limited[focal][focal] = true;
                for target in focal..params::N {
                    if (self.level_of[focal] as i32 - self.level_of[target] as i32).abs() > 1 {
                        self.network_limited[focal][focal] = true;
                    }
                }
            }
        }

        self.na = network_analyzer::NetworkAnalyzer::new();
    }

    /// Equivalent to private void initializeEntity().
    fn initialize_entity(&mut self) {
        // reality and realityBundleID
        for bundle in 0..params::M_OF_BUNDLE {
            let base_index = bundle * params::M_IN_BUNDLE;
            let end_index = (bundle + 1) * params::M_IN_BUNDLE;
            for m in base_index..end_index {
                if self.rng.gen::<bool>() {
                    self.reality[m] = true;
                } else {
                    self.reality[m] = false;
                }
                self.reality_bundle_id[m] = bundle;
            }
        }

        // beliefOf
        for focal in 0..params::N {
            for m in 0..params::M {
                self.belief_of[focal][m] = self.rng.gen::<bool>();
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

    /// Equivalent to void stepForward(int numFormation, int numBreak).
    pub fn step_forward_num_formation_break(&mut self, num_formation: usize, num_break: usize) {
        if params::DO_POST_REWIRING {
            if self.is_rewiring {
                if !self.is_random_rewiring {
                    self.do_rewiring(num_formation, num_break);
                } else {
                    self.do_random_rewiring(num_formation, num_break);
                }
            }
        }
        self.step_forward();
    }

    /// Equivalent to void stepForward(int tieTurnover).
    pub fn step_forward_tie_turnover(&mut self, tie_turnover: usize) {
        if params::DO_POST_REWIRING {
            if self.is_rewiring {
                if !self.is_random_rewiring {
                    self.do_rewiring(tie_turnover, tie_turnover);
                } else {
                    self.do_random_rewiring(tie_turnover, tie_turnover);
                }
            }
        }
        self.step_forward();
    }

    /// Equivalent to void stepForward().
    pub fn step_forward(&mut self) {
        self.do_learning();
        self.set_outcome();
        if self.turnover_rate > 0.0 {
            self.do_turnover();
        }
    }

    /// We remove the four instance variables from the class in Java,
    /// but we keep this method logic intact (now they are local).
    pub fn set_outcome(&mut self) {
        self.performance_avg = 0.0;
        self.na.set_network_metrics();

        self.average_path_length = self.na.get_average_path_length();
        self.network_efficiency = self.na.get_network_efficiency();
        self.global_clustering_watts_strogatz = self.na.get_global_clustering_watts_strogatz();
        self.overall_centralization = self.na.get_global_closeness_centralization();
        self.shortest_path_variance = self.na.get_shortest_path_variance();

        for focal in 0..params::N {
            self.performance_avg += self.performance_usize[focal] as f64;
        }
        self.performance_avg /= params::M_N as f64;
    }

    /// Equivalent to void setPreferenceScore().
    pub fn set_preference_score(&mut self) {
        self.preference_score = vec![vec![0.0; params::N]; params::N];
        self.preference_score_avg = vec![0.0; params::N];

        if self.is_network_closure {
            self.set_preference_score_raw_network_closure();
        } else if self.is_preferential_attachment {
            self.set_preference_score_raw_preferential_attachment();
        }
    }

    /// Equivalent to void setPreferenceScoreRawNetworkClosure().
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

    /// Equivalent to void setPreferenceScoreRawPreferentialAttachment().
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
        for shared in 0..params::N {
            if self.network[focal][shared] && self.network[target][shared] {
                preference_score += 1.0;
            }
        }
        let denom = cmp::max(self.degree[focal], self.degree[target]) as f64
            + if self.network[focal][target] { 0.0 } else { 1.0 };
        preference_score / denom
    }

    /// Equivalent to double getRewiringWeightPreferentialAttachment(int focal, int target).
    fn get_rewiring_weight_preferential_attachment(&self, focal: usize, target: usize) -> f64 {
        let df = self.degree[focal] as f64;
        let dt = self.degree[target] as f64;
        df.min(dt)
    }

    /// Equivalent to void doRewiring(int numFormation, int numBreak).
    fn do_rewiring(&mut self, num_formation: usize, num_break: usize) {
        if self.is_random_rewiring {
            self.do_random_rewiring(num_formation, num_break);
        } else {
            self.do_tie_break(num_break);
            self.do_tie_formation(num_formation);
        }
    }

    /// Equivalent to void doTieBreak(int numBreak).
    fn do_tie_break(&mut self, mut num_break: usize) {
        while num_break > 0 {
            let mut probability = vec![0.0; params::N_DYAD];
            let mut dyad2_cut_weight_max = f64::MIN_POSITIVE; // Java's Double.MIN_VALUE
            // Calculate rewiring weights
            for &d in self.iterator_dyad_index.iter() {
                let focal = self.map_dyad2d_index[d][0];
                let target = self.map_dyad2d_index[d][1];
                if self.network_informal[focal][target] {
                    let w = self.get_rewiring_weight(focal, target);
                    probability[d] = w;
                    if w > dyad2_cut_weight_max {
                        dyad2_cut_weight_max = w;
                    }
                }
            }
            // Convert to "largest becomes zero" style
            let mut probability_denominator = 0.0;
            for &d in self.iterator_dyad_index.iter() {
                if probability[d] != 0.0 {
                    probability[d] = dyad2_cut_weight_max - probability[d];
                    probability_denominator += probability[d];
                }
            }
            // If no edges had probability, shuffle and break randomly
            if probability_denominator == 0.0 {
                self.iterator_dyad_index.shuffle(&mut self.rng);
                for &d in self.iterator_dyad_index.iter() {
                    probability[d] = 1.0;
                }
                probability_denominator = 1.0;
            }
            // Choose
            let marker = self.rng.gen::<f64>();
            let mut probability_cum = 0.0;
            for &d in self.iterator_dyad_index.iter() {
                if probability[d] != 0.0 {
                    probability_cum += probability[d] / probability_denominator;
                    if probability_cum >= marker {
                        let focal = self.map_dyad2d_index[d][0];
                        let target = self.map_dyad2d_index[d][1];
                        self.network[focal][target] = false;
                        self.network[target][focal] = false;
                        self.network_informal[focal][target] = false;
                        self.network_informal[target][focal] = false;
                        self.degree[focal] -= 1;
                        self.degree_informal[focal] -= 1;
                        self.degree[target] -= 1;
                        self.degree_informal[target] -= 1;
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
            for &d in self.iterator_dyad_index.iter() {
                let focal = self.map_dyad2d_index[d][0];
                let target = self.map_dyad2d_index[d][1];
                if !self.network[focal][target]
                    && focal != target
                    && !self.network_limited[focal][target]
                {
                    probability[d] = self.get_rewiring_weight(focal, target);
                    probability_denominator += probability[d];
                }
            }
            if probability_denominator == 0.0 {
                self.iterator_dyad_index.shuffle(&mut self.rng);
                for &d in self.iterator_dyad_index.iter() {
                    probability[d] = 1.0;
                }
                probability_denominator = 1.0;
            }
            let marker = self.rng.gen::<f64>();
            let mut probability_cum = 0.0;
            for &d in self.iterator_dyad_index.iter() {
                if probability[d] != 0.0 {
                    probability_cum += probability[d] / probability_denominator;
                    if probability_cum >= marker {
                        let focal = self.map_dyad2d_index[d][0];
                        let target = self.map_dyad2d_index[d][1];
                        self.network[focal][target] = true;
                        self.network[target][focal] = true;
                        self.network_informal[focal][target] = true;
                        self.network_informal[target][focal] = true;
                        self.degree[focal] += 1;
                        self.degree_informal[focal] += 1;
                        self.degree[target] += 1;
                        self.degree_informal[target] += 1;
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
            self.iterator_dyad_index.shuffle(&mut self.rng);
            for &dyad in self.iterator_dyad_index.iter() {
                let focal = self.map_dyad2d_index[dyad][0];
                let target = self.map_dyad2d_index[dyad][1];
                if self.network_informal[focal][target] && num_break > 0 {
                    // Remove this informal tie
                    self.network[focal][target] = false;
                    self.network[target][focal] = false;
                    self.network_informal[focal][target] = false;
                    self.network_informal[target][focal] = false;
                    self.degree[focal] -= 1;
                    self.degree_informal[focal] -= 1;
                    self.degree[target] -= 1;
                    self.degree_informal[target] -= 1;
                    num_break -= 1;
                } else if num_formation > 0
                    && !self.network[focal][target]
                    && focal != target
                    && (self.degree_informal[focal] < params::INFORMAL_MAX_NUM
                        || self.degree_informal[target] < params::INFORMAL_MAX_NUM)
                    && !self.network_limited[focal][target]
                {
                    self.network[focal][target] = true;
                    self.network[target][focal] = true;
                    self.network_informal[focal][target] = true;
                    self.network_informal[target][focal] = true;
                    self.degree[focal] += 1;
                    self.degree_informal[focal] += 1;
                    self.degree[target] += 1;
                    self.degree_informal[target] += 1;
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

    /// Equivalent to void doLearning().
    fn do_learning(&mut self) {
        // We'll create a temporary buffer of beliefs
        let mut belief_of_buffer = vec![vec![false; params::M]; params::N];
        for focal in 0..params::N {
            // Copy current belief
            for m in 0..params::M {
                belief_of_buffer[focal][m] = self.belief_of[focal][m];
            }
            // Tally majority from better-performing neighbors
            let mut majority_opinion_count = vec![0; params::M];
            for target in 0..params::N {
                if self.network[focal][target] && self.performance_usize[target] > self.performance_usize[focal]
                {
                    for m in 0..params::M {
                        if self.belief_of[target][m] {
                            majority_opinion_count[m] += 1;
                        } else {
                            majority_opinion_count[m] -= 1;
                        }
                    }
                }
            }
            // Decide which bits to flip
            for m in 0..params::M {
                if majority_opinion_count[m] > 0 {
                    belief_of_buffer[focal][m] = true;
                } else if majority_opinion_count[m] < 0 {
                    belief_of_buffer[focal][m] = false;
                }
            }
        }
        // Apply with probability params::P_LEARNING
        for focal in 0..params::N {
            for m in 0..params::M {
                if self.belief_of[focal][m] != belief_of_buffer[focal][m] {
                    if self.rng.gen::<f64>() < params::P_LEARNING {
                        self.belief_of[focal][m] = belief_of_buffer[focal][m];
                    }
                }
            }
            self.set_performance_single(focal);
        }
    }

    /// Equivalent to int getPerformance(int focal).
    fn get_performance(&self, focal: usize) -> usize {
        let mut performance_now = 0;
        let belief_of_focal = &self.belief_of[focal];
        for bundle in 0..params::M_OF_BUNDLE {
            let start = bundle * params::M_IN_BUNDLE;
            let end = start + params::M_IN_BUNDLE;
            let mut match_all = true;
            for m in start..end {
                if belief_of_focal[m] != self.reality[m] {
                    match_all = false;
                    break;
                }
            }
            if match_all {
                // In Java, performanceNow += params::M_IN_BUNDLE
                performance_now += params::M_IN_BUNDLE as usize;
            }
        }
        performance_now
    }

    /// Equivalent to int getPerformanceS1(int focal).
    fn get_performance_s1(&self, focal: usize) -> usize {
        let mut performance_now = 0;
        for m in 0..params::M {
            if self.belief_of[focal][m] == self.reality[m] {
                performance_now += 1;
            }
        }
        performance_now
    }

    /// Equivalent to void setPerformance(int focal).
    fn set_performance_single(&mut self, focal: usize) {
        if params::M_IN_BUNDLE == 1 {
            self.performance_usize[focal] = self.get_performance_s1(focal);
        } else {
            self.performance_usize[focal] = self.get_performance(focal);
        }
    }

    /// Equivalent to void setPerformance().
    fn set_performance(&mut self) {
        for focal in 0..params::N {
            self.set_performance_single(focal);
        }
    }

    /// Equivalent to void printCSV(String fileName).
    pub fn print_csv(&self, file_name: &str) {
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
        for i in 0..params::N {
            if self.rng.gen::<f64>() < self.turnover_rate {
                // The individual i leaves; fill with new random beliefs
                for m in 0..params::M {
                    self.belief_of[m][params::M] = self.rng.gen::<bool>();
                }
                // Recompute performance for this individual
                self.set_performance_single(i);
            }
        }
    }

    /// doTurbulence(): each dimension of reality is flipped with probability turbulenceRate.
    pub fn do_turbulence(&mut self) {
        for m in 0..params::M {
            if self.rng.gen::<f64>() < self.turbulence_rate {
                self.reality[m] = !self.reality[m];
            }
        }
    }
}