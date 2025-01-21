// For parallel iteration
use rayon::prelude::*; 
use crate::{hdf5_manager, scenario::Scenario};
use crate::params;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{ArrayD, IxDyn};
use std::sync::{Arc, Mutex};

/// Manages the experiment, including running the experiment and processing results.
/// Modify as needed based on your experiment design.
pub struct ExperimentManager {
    pub r_perf_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_perf_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_perf_nr_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_perf_nr_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_perf_rr_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_perf_rr_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_perf_12_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_perf_12_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_perf_23_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_perf_23_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_perf_13_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_perf_13_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_clws_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_clws_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_clws_12_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_clws_12_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_clws_23_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_clws_23_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_clws_13_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_clws_13_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_clws_nr_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_clws_nr_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_clws_rr_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_clws_rr_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_cent_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_cent_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_cent_12_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_cent_12_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_cent_23_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_cent_23_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_cent_13_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_cent_13_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_cent_nr_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_cent_nr_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_cent_rr_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_cent_rr_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_effi_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_effi_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_effi_12_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_effi_12_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_effi_23_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_effi_23_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_effi_13_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_effi_13_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_effi_nr_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_effi_nr_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_effi_rr_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_effi_rr_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_sigm_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_sigm_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_sigm_12_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_sigm_12_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_sigm_23_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_sigm_23_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_sigm_13_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_sigm_13_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_sigm_nr_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_sigm_nr_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_sigm_rr_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_sigm_rr_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_omeg_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_omeg_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_omeg_12_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_omeg_12_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_omeg_23_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_omeg_23_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_omeg_13_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_omeg_13_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_omeg_nr_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_omeg_nr_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_omeg_rr_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_omeg_rr_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_spva_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_spva_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_spva_12_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_spva_12_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_spva_23_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_spva_23_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_spva_13_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_spva_13_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_spva_nr_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_spva_nr_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_spva_rr_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_spva_rr_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
}


impl ExperimentManager {

    pub fn new() -> Self{
        let shape = IxDyn(&params::RESULT_SHAPE);
        let shape_clone = shape.clone();
        ExperimentManager {
            r_perf_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_perf_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_perf_nr_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_perf_nr_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_perf_rr_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_perf_rr_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_perf_12_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_perf_12_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_perf_23_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_perf_23_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_perf_13_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_perf_13_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_clws_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_clws_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_clws_12_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_clws_12_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_clws_23_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_clws_23_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_clws_13_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_clws_13_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_clws_nr_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_clws_nr_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_clws_rr_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_clws_rr_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_cent_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_cent_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_cent_12_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_cent_12_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_cent_23_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_cent_23_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_cent_13_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_cent_13_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_cent_nr_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_cent_nr_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_cent_rr_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_cent_rr_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_effi_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_effi_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_effi_12_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_effi_12_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_effi_23_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_effi_23_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_effi_13_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_effi_13_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_effi_nr_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_effi_nr_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_effi_rr_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_effi_rr_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_sigm_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_sigm_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_sigm_12_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_sigm_12_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_sigm_23_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_sigm_23_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_sigm_13_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_sigm_13_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_sigm_nr_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_sigm_nr_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_sigm_rr_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_sigm_rr_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_omeg_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_omeg_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_omeg_12_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_omeg_12_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_omeg_23_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_omeg_23_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_omeg_13_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_omeg_13_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_omeg_nr_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_omeg_nr_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_omeg_rr_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_omeg_rr_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_spva_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_spva_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_spva_12_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_spva_12_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_spva_23_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_spva_23_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_spva_13_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_spva_13_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_spva_nr_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_spva_nr_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_spva_rr_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_spva_rr_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
        }
    }

    pub fn run_experiments(&mut self) {
        let tic = std::time::Instant::now().elapsed().as_millis();
        // Iterate over each combination in parallel
        let total_steps = params::PARAMS_INDEX_COMBINATIONS.get().unwrap().len() * params::ITERATION * params::TIME;
        let pb = ProgressBar::new(total_steps as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta_precise})")
                .expect("Failed to set progress bar template")
                .progress_chars("#>-"),
        );

        params::PARAMS_INDEX_COMBINATIONS.get().unwrap().into_par_iter().for_each(|(
            i_social_dynamics, 
            i_span, 
            i_enforcement, 
            i_turbulence, 
            i_turnover,
        )| {
            let turbulence_interval = params::TURBULENCE_INTERVAL[*i_turbulence];
            for _ in 0..params::ITERATION {
                // Create a new Scenario with the given parameters
                let mut scenario = Scenario::new(
                    *i_social_dynamics,
                    params::SPAN[*i_span],
                    params::ENFORCEMENT[*i_enforcement],
                    params::TURBULENCE_RATE[*i_turbulence],
                    params::TURNOVER_RATE[*i_turnover],
                );
                
                let mut scenario_random_rewiring = scenario.get_clone();
                scenario_random_rewiring.set_network_params(true, true);

                let mut scenario_no_rewiring = scenario.get_clone();
                scenario_no_rewiring.set_network_params(false, false);

                scenario.do_rewiring(params::INFORMAL_INITIAL_NUM, 0); // Systematically formed
                scenario_random_rewiring.do_rewiring(params::INFORMAL_INITIAL_NUM, 0); // Randomly formed

                for t in 0..params::TIME {
                    scenario.step_forward();
                    scenario_random_rewiring.step_forward();
                    scenario_no_rewiring.step_forward();
                    
                    if t % turbulence_interval == 0 {
                        scenario.do_turbulence();
                        scenario_random_rewiring.do_turbulence();
                        scenario_no_rewiring.do_turbulence();
                    }

                    self.r_perf_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.performance_avg;
                    self.r_perf_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.performance_avg.powi(2);
                    self.r_perf_rr_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_random_rewiring.performance_avg;
                    self.r_perf_rr_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_random_rewiring.performance_avg.powi(2);
                    self.r_perf_nr_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_no_rewiring.performance_avg;
                    self.r_perf_nr_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_no_rewiring.performance_avg.powi(2);
                    self.r_perf_12_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.performance_avg - scenario_random_rewiring.performance_avg;
                    self.r_perf_12_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += (scenario.performance_avg - scenario_random_rewiring.performance_avg).powi(2);
                    self.r_perf_23_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_random_rewiring.performance_avg - scenario_no_rewiring.performance_avg;
                    self.r_perf_23_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += (scenario_random_rewiring.performance_avg - scenario_no_rewiring.performance_avg).powi(2);
                    self.r_perf_13_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.performance_avg - scenario_no_rewiring.performance_avg;
                    self.r_perf_13_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += (scenario.performance_avg - scenario_no_rewiring.performance_avg).powi(2);
                    self.r_clws_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.global_clustering_watts_strogatz;
                    self.r_clws_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.global_clustering_watts_strogatz.powi(2);
                    self.r_clws_rr_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_random_rewiring.global_clustering_watts_strogatz;
                    self.r_clws_rr_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_random_rewiring.global_clustering_watts_strogatz.powi(2);
                    self.r_clws_nr_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_no_rewiring.global_clustering_watts_strogatz;
                    self.r_clws_nr_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_no_rewiring.global_clustering_watts_strogatz.powi(2);
                    self.r_clws_12_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.global_clustering_watts_strogatz - scenario_random_rewiring.global_clustering_watts_strogatz;
                    self.r_clws_12_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += (scenario.global_clustering_watts_strogatz - scenario_random_rewiring.global_clustering_watts_strogatz).powi(2);
                    self.r_clws_23_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_random_rewiring.global_clustering_watts_strogatz - scenario_no_rewiring.global_clustering_watts_strogatz;
                    self.r_clws_23_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += (scenario_random_rewiring.global_clustering_watts_strogatz - scenario_no_rewiring.global_clustering_watts_strogatz).powi(2);
                    self.r_clws_13_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.global_clustering_watts_strogatz - scenario_no_rewiring.global_clustering_watts_strogatz;
                    self.r_clws_13_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += (scenario.global_clustering_watts_strogatz - scenario_no_rewiring.global_clustering_watts_strogatz).powi(2);
                    self.r_cent_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.overall_centralization;
                    self.r_cent_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.overall_centralization.powi(2);
                    self.r_cent_rr_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_random_rewiring.overall_centralization;
                    self.r_cent_rr_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_random_rewiring.overall_centralization.powi(2);
                    self.r_cent_nr_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_no_rewiring.overall_centralization;
                    self.r_cent_nr_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_no_rewiring.overall_centralization.powi(2);
                    self.r_cent_12_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.overall_centralization - scenario_random_rewiring.overall_centralization;
                    self.r_cent_12_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += (scenario.overall_centralization - scenario_random_rewiring.overall_centralization).powi(2);
                    self.r_cent_23_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_random_rewiring.overall_centralization - scenario_no_rewiring.overall_centralization;
                    self.r_cent_23_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += (scenario_random_rewiring.overall_centralization - scenario_no_rewiring.overall_centralization).powi(2);
                    self.r_cent_13_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.overall_centralization - scenario_no_rewiring.overall_centralization;
                    self.r_cent_13_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += (scenario.overall_centralization - scenario_no_rewiring.overall_centralization).powi(2);
                    self.r_effi_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.network_efficiency;
                    self.r_effi_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.network_efficiency.powi(2);
                    self.r_effi_rr_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_random_rewiring.network_efficiency;
                    self.r_effi_rr_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_random_rewiring.network_efficiency.powi(2);
                    self.r_effi_nr_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_no_rewiring.network_efficiency;
                    self.r_effi_nr_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_no_rewiring.network_efficiency.powi(2);
                    self.r_effi_12_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.network_efficiency - scenario_random_rewiring.network_efficiency;
                    self.r_effi_12_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += (scenario.network_efficiency - scenario_random_rewiring.network_efficiency).powi(2);
                    self.r_effi_23_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_random_rewiring.network_efficiency - scenario_no_rewiring.network_efficiency;
                    self.r_effi_23_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += (scenario_random_rewiring.network_efficiency - scenario_no_rewiring.network_efficiency).powi(2);
                    self.r_effi_13_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.network_efficiency - scenario_no_rewiring.network_efficiency;
                    self.r_effi_13_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += (scenario.network_efficiency - scenario_no_rewiring.network_efficiency).powi(2);
                    self.r_sigm_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.sigma;
                    self.r_sigm_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.sigma.powi(2);
                    self.r_sigm_rr_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_random_rewiring.sigma;
                    self.r_sigm_rr_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_random_rewiring.sigma.powi(2);
                    self.r_sigm_nr_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_no_rewiring.sigma;
                    self.r_sigm_nr_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_no_rewiring.sigma.powi(2);
                    self.r_sigm_12_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.sigma - scenario_random_rewiring.sigma;
                    self.r_sigm_12_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += (scenario.sigma - scenario_random_rewiring.sigma).powi(2);
                    self.r_sigm_23_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_random_rewiring.sigma - scenario_no_rewiring.sigma;
                    self.r_sigm_23_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += (scenario_random_rewiring.sigma - scenario_no_rewiring.sigma).powi(2);
                    self.r_sigm_13_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.sigma - scenario_no_rewiring.sigma;
                    self.r_sigm_13_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += (scenario.sigma - scenario_no_rewiring.sigma).powi(2);
                    self.r_omeg_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.omega;
                    self.r_omeg_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.omega.powi(2);
                    self.r_omeg_rr_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_random_rewiring.omega;
                    self.r_omeg_rr_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_random_rewiring.omega.powi(2);
                    self.r_omeg_nr_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_no_rewiring.omega;
                    self.r_omeg_nr_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_no_rewiring.omega.powi(2);
                    self.r_omeg_12_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.omega - scenario_random_rewiring.omega;
                    self.r_omeg_12_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += (scenario.omega - scenario_random_rewiring.omega).powi(2);
                    self.r_omeg_23_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_random_rewiring.omega - scenario_no_rewiring.omega;
                    self.r_omeg_23_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += (scenario_random_rewiring.omega - scenario_no_rewiring.omega).powi(2);
                    self.r_omeg_13_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.omega - scenario_no_rewiring.omega;
                    self.r_omeg_13_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += (scenario.omega - scenario_no_rewiring.omega).powi(2);
                    self.r_spva_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.shortest_path_variance;
                    self.r_spva_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.shortest_path_variance.powi(2);
                    self.r_spva_rr_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_random_rewiring.shortest_path_variance;
                    self.r_spva_rr_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_random_rewiring.shortest_path_variance.powi(2);
                    self.r_spva_nr_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_no_rewiring.shortest_path_variance;
                    self.r_spva_nr_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_no_rewiring.shortest_path_variance.powi(2);
                    self.r_spva_12_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.shortest_path_variance - scenario_random_rewiring.shortest_path_variance;
                    self.r_spva_12_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += (scenario.shortest_path_variance - scenario_random_rewiring.shortest_path_variance).powi(2);
                    self.r_spva_23_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario_random_rewiring.shortest_path_variance - scenario_no_rewiring.shortest_path_variance;
                    self.r_spva_23_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += (scenario_random_rewiring.shortest_path_variance - scenario_no_rewiring.shortest_path_variance).powi(2);
                    self.r_spva_13_avg.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += scenario.shortest_path_variance - scenario_no_rewiring.shortest_path_variance;
                    self.r_spva_13_std.lock().unwrap()[[*i_social_dynamics, *i_span, *i_enforcement, *i_turbulence, *i_turnover, t]] += (scenario.shortest_path_variance - scenario_no_rewiring.shortest_path_variance).powi(2);

                    println!("social_dynamics: {}, span: {}, enforcement: {}, turbulence_rate: {}, turnover_rate: {}, time: {}, performance_avg: {}",
                    *i_social_dynamics, 
                    *i_span, 
                    *i_enforcement, 
                    *i_turbulence, 
                    *i_turnover, 
                    t,
                    scenario.performance_avg
                );

                pb.inc(1); // Increment the progress bar


                }

            }
        });
        pb.finish_with_message("Done!");

    }
 
}