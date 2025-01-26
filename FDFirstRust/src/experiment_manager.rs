use rayon::prelude::*; 
use ndarray::{ArrayD, IxDyn};
use std::sync::{Arc, Mutex};
use indicatif::{ProgressBar, ProgressStyle};
use chrono::Local;
use crate::params;
use crate::scenario::Scenario;

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
    pub r_tria_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_tria_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_tria_12_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_tria_12_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_tria_23_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_tria_23_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_tria_13_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_tria_13_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_tria_nr_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_tria_nr_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_tria_rr_avg: Arc::<Mutex<ndarray::ArrayD<f64>>>,
    pub r_tria_rr_std: Arc::<Mutex<ndarray::ArrayD<f64>>>,
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
        let shape: ndarray::Dim<ndarray::IxDynImpl> = IxDyn(&params::RESULT_SHAPE);
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
            r_tria_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_tria_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_tria_12_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_tria_12_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_tria_23_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_tria_23_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_tria_13_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_tria_13_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_tria_nr_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_tria_nr_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_tria_rr_avg: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
            r_tria_rr_std: Arc::new(Mutex::new(ArrayD::zeros(shape.clone()))),
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
        // Iterate over each combination in parallel
        let length_combination = params::PARAMS_INDEX_COMBINATIONS.get().unwrap().len() * params::ITERATION * params::TIME;
        let pb = ProgressBar::new(length_combination as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta_precise})")
                .expect("Failed to set progress bar template")
                .progress_chars("#>-"),
        );
        println!(
            "{}\tInitiated: Running Experiments - Target file: {}",
            Local::now().format("%Y-%m-%d %H:%M:%S"),
            (params::PARAM_STRING).clone(),
        );

        //TODO: Create value list within each combination, then save it to ndarray after iteration. This way reduces the number of locks.
        params::PARAMS_INDEX_COMBINATIONS.get().unwrap().into_par_iter().for_each(
            |(
            i_social_dynamics, 
            i_span, 
            i_enforcement, 
            i_turbulence, 
            i_turnover)| {
            let indices = vec![
                *i_social_dynamics,
                *i_span,
                *i_enforcement,
                *i_turbulence,
                *i_turnover,
            ];
            let span = params::SPAN[*i_span];
            let enforcement = params::ENFORCEMENT[*i_enforcement];
            let turbulence_rate = params::TURBULENCE_RATE[*i_turbulence];
            let turnover_rate = params::TURNOVER_RATE[*i_turnover];
            let turbulence_interval = params::TURBULENCE_INTERVAL[*i_turbulence];

            let mut local_perf = OutcomeVariable::new();
            let mut local_perf_rr = OutcomeVariable::new();
            let mut local_perf_nr = OutcomeVariable::new();
            let mut local_perf_12 = OutcomeVariable::new();
            let mut local_perf_23 = OutcomeVariable::new();
            let mut local_perf_13 = OutcomeVariable::new();

            let mut local_clws = OutcomeVariable::new();
            let mut local_clws_rr = OutcomeVariable::new();
            let mut local_clws_nr = OutcomeVariable::new();
            let mut local_clws_12 = OutcomeVariable::new();
            let mut local_clws_23 = OutcomeVariable::new();
            let mut local_clws_13 = OutcomeVariable::new();
            
            let mut local_cent = OutcomeVariable::new();
            let mut local_cent_rr = OutcomeVariable::new();
            let mut local_cent_nr = OutcomeVariable::new();
            let mut local_cent_12 = OutcomeVariable::new();
            let mut local_cent_23 = OutcomeVariable::new();
            let mut local_cent_13 = OutcomeVariable::new();
            
            let mut local_tria = OutcomeVariable::new();
            let mut local_tria_rr = OutcomeVariable::new();
            let mut local_tria_nr = OutcomeVariable::new();
            let mut local_tria_12 = OutcomeVariable::new();
            let mut local_tria_23 = OutcomeVariable::new();
            let mut local_tria_13 = OutcomeVariable::new();

            let mut local_effi = OutcomeVariable::new();
            let mut local_effi_rr = OutcomeVariable::new();
            let mut local_effi_nr = OutcomeVariable::new();
            let mut local_effi_12 = OutcomeVariable::new();
            let mut local_effi_23 = OutcomeVariable::new();
            let mut local_effi_13 = OutcomeVariable::new();

            let mut local_sigm = OutcomeVariable::new();
            let mut local_sigm_rr = OutcomeVariable::new();
            let mut local_sigm_nr = OutcomeVariable::new();
            let mut local_sigm_12 = OutcomeVariable::new();
            let mut local_sigm_23 = OutcomeVariable::new();
            let mut local_sigm_13 = OutcomeVariable::new();

            let mut local_omeg = OutcomeVariable::new();
            let mut local_omeg_rr = OutcomeVariable::new();
            let mut local_omeg_nr = OutcomeVariable::new();
            let mut local_omeg_12 = OutcomeVariable::new();
            let mut local_omeg_23 = OutcomeVariable::new();
            let mut local_omeg_13 = OutcomeVariable::new();
            
            let mut local_spva = OutcomeVariable::new();
            let mut local_spva_rr = OutcomeVariable::new();
            let mut local_spva_nr = OutcomeVariable::new();
            let mut local_spva_12 = OutcomeVariable::new();
            let mut local_spva_23 = OutcomeVariable::new();
            let mut local_spva_13 = OutcomeVariable::new();

            for _ in 0..params::ITERATION {
                // Create a new Scenario with the given parameters
                let mut scenario = Scenario::new(
                    *i_social_dynamics,
                    span,
                    enforcement,
                    turbulence_rate,
                    turnover_rate,
                );
                
                let mut scenario_random_rewiring = scenario.get_clone();
                scenario_random_rewiring.set_network_params(true, true);
                let mut scenario_no_rewiring = scenario.get_clone();
                scenario_no_rewiring.set_network_params(false, false);
                scenario.do_rewiring(params::INFORMAL_INITIAL_NUM, 0); // Systematically formed
                scenario_random_rewiring.do_rewiring(params::INFORMAL_INITIAL_NUM, 0); // Randomly formed

                for t in 0..params::TIME {
                    local_perf.accumulate(t, scenario.performance_avg);
                    local_perf_rr.accumulate(t, scenario_random_rewiring.performance_avg);
                    local_perf_nr.accumulate(t, scenario_no_rewiring.performance_avg);
                    local_perf_12.accumulate(t, scenario.performance_avg - scenario_random_rewiring.performance_avg);
                    local_perf_23.accumulate(t, scenario_random_rewiring.performance_avg - scenario_no_rewiring.performance_avg);
                    local_perf_13.accumulate(t, scenario.performance_avg - scenario_no_rewiring.performance_avg);

                    local_clws.accumulate(t, scenario.global_clustering_watts_strogatz);
                    local_clws_rr.accumulate(t, scenario_random_rewiring.global_clustering_watts_strogatz);
                    local_clws_nr.accumulate(t, scenario_no_rewiring.global_clustering_watts_strogatz);
                    local_clws_12.accumulate(t, scenario.global_clustering_watts_strogatz - scenario_random_rewiring.global_clustering_watts_strogatz);
                    local_clws_23.accumulate(t, scenario_random_rewiring.global_clustering_watts_strogatz - scenario_no_rewiring.global_clustering_watts_strogatz);
                    local_clws_13.accumulate(t, scenario.global_clustering_watts_strogatz - scenario_no_rewiring.global_clustering_watts_strogatz);

                    local_cent.accumulate(t, scenario.closeness_centralization);
                    local_cent_rr.accumulate(t, scenario_random_rewiring.closeness_centralization);
                    local_cent_nr.accumulate(t, scenario_no_rewiring.closeness_centralization);
                    local_cent_12.accumulate(t, scenario.closeness_centralization - scenario_random_rewiring.closeness_centralization);
                    local_cent_23.accumulate(t, scenario_random_rewiring.closeness_centralization - scenario_no_rewiring.closeness_centralization);
                    local_cent_13.accumulate(t, scenario.closeness_centralization - scenario_no_rewiring.closeness_centralization);

                    local_tria.accumulate(t, scenario.triadic_centralization);
                    local_tria_rr.accumulate(t, scenario_random_rewiring.triadic_centralization);
                    local_tria_nr.accumulate(t, scenario_no_rewiring.triadic_centralization);
                    local_tria_12.accumulate(t, scenario.triadic_centralization - scenario_random_rewiring.triadic_centralization);
                    local_tria_23.accumulate(t, scenario_random_rewiring.triadic_centralization - scenario_no_rewiring.triadic_centralization);
                    local_tria_13.accumulate(t, scenario.triadic_centralization - scenario_no_rewiring.triadic_centralization);

                    local_spva.accumulate(t, scenario.shortest_path_variance);
                    local_spva_rr.accumulate(t, scenario_random_rewiring.shortest_path_variance);
                    local_spva_nr.accumulate(t, scenario_no_rewiring.shortest_path_variance);
                    local_spva_12.accumulate(t, scenario.shortest_path_variance - scenario_random_rewiring.shortest_path_variance);
                    local_spva_23.accumulate(t, scenario_random_rewiring.shortest_path_variance - scenario_no_rewiring.shortest_path_variance);
                    local_spva_13.accumulate(t, scenario.shortest_path_variance - scenario_no_rewiring.shortest_path_variance);
                    
                    local_effi.accumulate(t, scenario.network_efficiency);
                    local_effi_rr.accumulate(t, scenario_random_rewiring.network_efficiency);
                    local_effi_nr.accumulate(t, scenario_no_rewiring.network_efficiency);
                    local_effi_12.accumulate(t, scenario.network_efficiency - scenario_random_rewiring.network_efficiency);
                    local_effi_23.accumulate(t, scenario_random_rewiring.network_efficiency - scenario_no_rewiring.network_efficiency);
                    local_effi_13.accumulate(t, scenario.network_efficiency - scenario_no_rewiring.network_efficiency);

                    local_sigm.accumulate(t, scenario.omega);
                    local_sigm_rr.accumulate(t, scenario_random_rewiring.omega);
                    local_sigm_nr.accumulate(t, scenario_no_rewiring.omega);
                    local_sigm_12.accumulate(t, scenario.omega - scenario_random_rewiring.omega);
                    local_sigm_23.accumulate(t, scenario_random_rewiring.omega - scenario_no_rewiring.omega);
                    local_sigm_13.accumulate(t, scenario.omega - scenario_no_rewiring.omega);
                    
                    local_omeg.accumulate(t, scenario.omega);
                    local_omeg_rr.accumulate(t, scenario_random_rewiring.omega);
                    local_omeg_nr.accumulate(t, scenario_no_rewiring.omega);
                    local_omeg_12.accumulate(t, scenario.omega - scenario_random_rewiring.omega);
                    local_omeg_23.accumulate(t, scenario_random_rewiring.omega - scenario_no_rewiring.omega);
                    local_omeg_13.accumulate(t, scenario.omega - scenario_no_rewiring.omega);


                    scenario.step_forward();
                    scenario_random_rewiring.step_forward();
                    scenario_no_rewiring.step_forward();
                    if t % turbulence_interval == 0 {
                        scenario.do_turbulence();
                        scenario_random_rewiring.do_turbulence();
                        scenario_no_rewiring.do_turbulence();
                    }
                    
                    pb.inc(1); // Increment the progress bar
                }
            }
            
            local_perf.finalize();
            local_perf_rr.finalize();
            local_perf_nr.finalize();
            local_perf_12.finalize();
            local_perf_23.finalize();
            local_perf_13.finalize();
            
            local_clws.finalize();
            local_clws_rr.finalize();
            local_clws_nr.finalize();
            local_clws_12.finalize();
            local_clws_23.finalize();
            local_clws_13.finalize();
            
            local_cent.finalize();
            local_cent_rr.finalize();
            local_cent_nr.finalize();
            local_cent_12.finalize();
            local_cent_23.finalize();
            local_cent_13.finalize();
            
            local_tria.finalize();
            local_tria_rr.finalize();
            local_tria_nr.finalize();
            local_tria_12.finalize();
            local_tria_23.finalize();
            local_tria_13.finalize();
            
            local_spva.finalize();
            local_spva_rr.finalize();
            local_spva_nr.finalize();
            local_spva_12.finalize();
            local_spva_23.finalize();
            local_spva_13.finalize();
            
            local_effi.finalize();
            local_effi_rr.finalize();
            local_effi_nr.finalize();
            local_effi_12.finalize();
            local_effi_23.finalize();
            local_effi_13.finalize();
            
            local_sigm.finalize();
            local_sigm_rr.finalize();
            local_sigm_nr.finalize();
            local_sigm_12.finalize();
            local_sigm_23.finalize();
            local_sigm_13.finalize();
            
            local_omeg.finalize();
            local_omeg_rr.finalize();
            local_omeg_nr.finalize();
            local_omeg_12.finalize();
            local_omeg_23.finalize();
            local_omeg_13.finalize();
            
            for t in 0..params::TIME {
                let mut indices_t = indices.clone();
                indices_t.push(t);
                let ix_dyn = IxDyn(&indices_t);
                self.r_perf_avg.lock().unwrap()[&ix_dyn] = local_perf.avg[t];
                self.r_perf_std.lock().unwrap()[&ix_dyn] = local_perf.std[t];
                self.r_perf_rr_avg.lock().unwrap()[&ix_dyn] = local_perf_rr.avg[t];
                self.r_perf_rr_std.lock().unwrap()[&ix_dyn] = local_perf_rr.std[t];
                self.r_perf_nr_avg.lock().unwrap()[&ix_dyn] = local_perf_nr.avg[t];
                self.r_perf_nr_std.lock().unwrap()[&ix_dyn] = local_perf_nr.std[t];
                self.r_perf_12_avg.lock().unwrap()[&ix_dyn] = local_perf_12.avg[t];
                self.r_perf_12_std.lock().unwrap()[&ix_dyn] = local_perf_12.std[t];
                self.r_perf_23_avg.lock().unwrap()[&ix_dyn] = local_perf_23.avg[t];
                self.r_perf_23_std.lock().unwrap()[&ix_dyn] = local_perf_23.std[t];
                self.r_perf_13_avg.lock().unwrap()[&ix_dyn] = local_perf_13.avg[t];
                self.r_perf_13_std.lock().unwrap()[&ix_dyn] = local_perf_13.std[t];
                
                self.r_clws_avg.lock().unwrap()[&ix_dyn] = local_clws.avg[t];
                self.r_clws_std.lock().unwrap()[&ix_dyn] = local_clws.std[t];
                self.r_clws_rr_avg.lock().unwrap()[&ix_dyn] = local_clws_rr.avg[t];
                self.r_clws_rr_std.lock().unwrap()[&ix_dyn] = local_clws_rr.std[t];
                self.r_clws_nr_avg.lock().unwrap()[&ix_dyn] = local_clws_nr.avg[t];
                self.r_clws_nr_std.lock().unwrap()[&ix_dyn] = local_clws_nr.std[t];
                self.r_clws_12_avg.lock().unwrap()[&ix_dyn] = local_clws_12.avg[t];
                self.r_clws_12_std.lock().unwrap()[&ix_dyn] = local_clws_12.std[t];
                self.r_clws_23_avg.lock().unwrap()[&ix_dyn] = local_clws_23.avg[t];
                self.r_clws_23_std.lock().unwrap()[&ix_dyn] = local_clws_23.std[t];
                self.r_clws_13_avg.lock().unwrap()[&ix_dyn] = local_clws_13.avg[t];
                self.r_clws_13_std.lock().unwrap()[&ix_dyn] = local_clws_13.std[t];
                
                self.r_cent_avg.lock().unwrap()[&ix_dyn] = local_cent.avg[t];
                self.r_cent_std.lock().unwrap()[&ix_dyn] = local_cent.std[t];
                self.r_cent_rr_avg.lock().unwrap()[&ix_dyn] = local_cent_rr.avg[t];
                self.r_cent_rr_std.lock().unwrap()[&ix_dyn] = local_cent_rr.std[t];
                self.r_cent_nr_avg.lock().unwrap()[&ix_dyn] = local_cent_nr.avg[t];
                self.r_cent_nr_std.lock().unwrap()[&ix_dyn] = local_cent_nr.std[t];
                self.r_cent_12_avg.lock().unwrap()[&ix_dyn] = local_cent_12.avg[t];
                self.r_cent_12_std.lock().unwrap()[&ix_dyn] = local_cent_12.std[t];
                self.r_cent_23_avg.lock().unwrap()[&ix_dyn] = local_cent_23.avg[t];
                self.r_cent_23_std.lock().unwrap()[&ix_dyn] = local_cent_23.std[t];
                self.r_cent_13_avg.lock().unwrap()[&ix_dyn] = local_cent_13.avg[t];
                self.r_cent_13_std.lock().unwrap()[&ix_dyn] = local_cent_13.std[t];
                
                self.r_tria_avg.lock().unwrap()[&ix_dyn] = local_tria.avg[t];
                self.r_tria_std.lock().unwrap()[&ix_dyn] = local_tria.std[t];
                self.r_tria_rr_avg.lock().unwrap()[&ix_dyn] = local_tria_rr.avg[t];
                self.r_tria_rr_std.lock().unwrap()[&ix_dyn] = local_tria_rr.std[t];
                self.r_tria_nr_avg.lock().unwrap()[&ix_dyn] = local_tria_nr.avg[t];
                self.r_tria_nr_std.lock().unwrap()[&ix_dyn] = local_tria_nr.std[t];
                self.r_tria_12_avg.lock().unwrap()[&ix_dyn] = local_tria_12.avg[t];
                self.r_tria_12_std.lock().unwrap()[&ix_dyn] = local_tria_12.std[t];
                self.r_tria_23_avg.lock().unwrap()[&ix_dyn] = local_tria_23.avg[t];
                self.r_tria_23_std.lock().unwrap()[&ix_dyn] = local_tria_23.std[t];
                self.r_tria_13_avg.lock().unwrap()[&ix_dyn] = local_tria_13.avg[t];
                self.r_tria_13_std.lock().unwrap()[&ix_dyn] = local_tria_13.std[t];
                
                self.r_spva_avg.lock().unwrap()[&ix_dyn] = local_spva.avg[t];
                self.r_spva_std.lock().unwrap()[&ix_dyn] = local_spva.std[t];
                self.r_spva_rr_avg.lock().unwrap()[&ix_dyn] = local_spva_rr.avg[t];
                self.r_spva_rr_std.lock().unwrap()[&ix_dyn] = local_spva_rr.std[t];
                self.r_spva_nr_avg.lock().unwrap()[&ix_dyn] = local_spva_nr.avg[t];
                self.r_spva_nr_std.lock().unwrap()[&ix_dyn] = local_spva_nr.std[t];
                self.r_spva_12_avg.lock().unwrap()[&ix_dyn] = local_spva_12.avg[t];
                self.r_spva_12_std.lock().unwrap()[&ix_dyn] = local_spva_12.std[t];
                self.r_spva_23_avg.lock().unwrap()[&ix_dyn] = local_spva_23.avg[t];
                self.r_spva_23_std.lock().unwrap()[&ix_dyn] = local_spva_23.std[t];
                self.r_spva_13_avg.lock().unwrap()[&ix_dyn] = local_spva_13.avg[t];
                self.r_spva_13_std.lock().unwrap()[&ix_dyn] = local_spva_13.std[t];
                
                self.r_effi_avg.lock().unwrap()[&ix_dyn] = local_effi.avg[t];
                self.r_effi_std.lock().unwrap()[&ix_dyn] = local_effi.std[t];
                self.r_effi_rr_avg.lock().unwrap()[&ix_dyn] = local_effi_rr.avg[t];
                self.r_effi_rr_std.lock().unwrap()[&ix_dyn] = local_effi_rr.std[t];
                self.r_effi_nr_avg.lock().unwrap()[&ix_dyn] = local_effi_nr.avg[t];
                self.r_effi_nr_std.lock().unwrap()[&ix_dyn] = local_effi_nr.std[t];
                self.r_effi_12_avg.lock().unwrap()[&ix_dyn] = local_effi_12.avg[t];
                self.r_effi_12_std.lock().unwrap()[&ix_dyn] = local_effi_12.std[t];
                self.r_effi_23_avg.lock().unwrap()[&ix_dyn] = local_effi_23.avg[t];
                self.r_effi_23_std.lock().unwrap()[&ix_dyn] = local_effi_23.std[t];
                self.r_effi_13_avg.lock().unwrap()[&ix_dyn] = local_effi_13.avg[t];
                self.r_effi_13_std.lock().unwrap()[&ix_dyn] = local_effi_13.std[t];
                
                self.r_sigm_avg.lock().unwrap()[&ix_dyn] = local_sigm.avg[t];
                self.r_sigm_std.lock().unwrap()[&ix_dyn] = local_sigm.std[t];
                self.r_sigm_rr_avg.lock().unwrap()[&ix_dyn] = local_sigm_rr.avg[t];
                self.r_sigm_rr_std.lock().unwrap()[&ix_dyn] = local_sigm_rr.std[t];
                self.r_sigm_nr_avg.lock().unwrap()[&ix_dyn] = local_sigm_nr.avg[t];
                self.r_sigm_nr_std.lock().unwrap()[&ix_dyn] = local_sigm_nr.std[t];
                self.r_sigm_12_avg.lock().unwrap()[&ix_dyn] = local_sigm_12.avg[t];
                self.r_sigm_12_std.lock().unwrap()[&ix_dyn] = local_sigm_12.std[t];
                self.r_sigm_23_avg.lock().unwrap()[&ix_dyn] = local_sigm_23.avg[t];
                self.r_sigm_23_std.lock().unwrap()[&ix_dyn] = local_sigm_23.std[t];
                self.r_sigm_13_avg.lock().unwrap()[&ix_dyn] = local_sigm_13.avg[t];
                self.r_sigm_13_std.lock().unwrap()[&ix_dyn] = local_sigm_13.std[t];

                self.r_omeg_avg.lock().unwrap()[&ix_dyn] = local_omeg.avg[t];
                self.r_omeg_std.lock().unwrap()[&ix_dyn] = local_omeg.std[t];
                self.r_omeg_rr_avg.lock().unwrap()[&ix_dyn] = local_omeg_rr.avg[t];
                self.r_omeg_rr_std.lock().unwrap()[&ix_dyn] = local_omeg_rr.std[t];
                self.r_omeg_nr_avg.lock().unwrap()[&ix_dyn] = local_omeg_nr.avg[t];
                self.r_omeg_nr_std.lock().unwrap()[&ix_dyn] = local_omeg_nr.std[t];
                self.r_omeg_12_avg.lock().unwrap()[&ix_dyn] = local_omeg_12.avg[t];
                self.r_omeg_12_std.lock().unwrap()[&ix_dyn] = local_omeg_12.std[t];
                self.r_omeg_23_avg.lock().unwrap()[&ix_dyn] = local_omeg_23.avg[t];
                self.r_omeg_23_std.lock().unwrap()[&ix_dyn] = local_omeg_23.std[t];
                self.r_omeg_13_avg.lock().unwrap()[&ix_dyn] = local_omeg_13.avg[t];
                self.r_omeg_13_std.lock().unwrap()[&ix_dyn] = local_omeg_13.std[t];
            }
            },
        );
        pb.finish_with_message("Done!");
    }

    pub fn sample_network_csv(&self){
                // Iterate over each combination in parallel
                let length_combination = params::PARAMS_INDEX_COMBINATIONS.get().unwrap().len();
                let pb = ProgressBar::new(length_combination as u64);
                pb.set_style(
                    ProgressStyle::default_bar()
                        .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta_precise})")
                        .expect("Failed to set progress bar template")
                        .progress_chars("#>-"),
                );
                println!(
                    "{}\tInitiated: Sampling Network in *.csv for {} combinations",
                    Local::now().format("%Y-%m-%d %H:%M:%S"),
                    length_combination
                );
                
                params::PARAMS_INDEX_COMBINATIONS.get().unwrap().into_par_iter().for_each(|(
                    i_social_dynamics, 
                    i_span, 
                    i_enforcement, 
                    i_turbulence, 
                    i_turnover,
                )| {
                    // Create a new Scenario with the given parameters
                    let span = params::SPAN[*i_span];
                    let enforcement = params::ENFORCEMENT[*i_enforcement];
                    let turbulence_rate = params::TURBULENCE_RATE[*i_turbulence];
                    let turnover_rate = params::TURNOVER_RATE[*i_turnover];
                    let turbulence_interval = params::TURBULENCE_INTERVAL[*i_turbulence];

                    let mut scenario = Scenario::new(
                        *i_social_dynamics,
                        span,
                        enforcement,
                        turbulence_rate,
                        turnover_rate,
                    );
                    
                    let mut scenario_random_rewiring = scenario.get_clone();
                    scenario_random_rewiring.set_network_params(true, true);
                    let mut scenario_no_rewiring = scenario.get_clone();
                    scenario_no_rewiring.set_network_params(false, false);
                    scenario.do_rewiring(params::INFORMAL_INITIAL_NUM, 0); // Systematically formed
                    scenario_random_rewiring.do_rewiring(params::INFORMAL_INITIAL_NUM, 0); // Randomly formed
    
                    let file_name_network_csv = format!("{}s{}e{}ptb{}itb{}ptn{}.csv", if *i_social_dynamics==0 {"NetCl"} else {"PrfAt"}, span, enforcement, turbulence_rate, turbulence_interval, turnover_rate);
                    let path_network_csv = (params::PARAM_STRING).clone();
                    scenario.export_network_csv(format!("{}/{}_{}_t0", &path_network_csv, "sc", &file_name_network_csv).as_str());
                    scenario_random_rewiring.export_network_csv(format!("{}/{}_{}_t0", &path_network_csv, "rr", &file_name_network_csv).as_str());
                    scenario_no_rewiring.export_network_csv(format!("{}/{}_{}_t0", &path_network_csv, "nr", &file_name_network_csv).as_str());
                    
                    for t in 0..params::TIME {
                        scenario.step_forward();
                        scenario_random_rewiring.step_forward();
                        scenario_no_rewiring.step_forward();
                        if t % turbulence_interval == 0 {
                            scenario.do_turbulence();
                            scenario_random_rewiring.do_turbulence();
                            scenario_no_rewiring.do_turbulence();
                        }
                    }

                    scenario.export_network_csv(format!("{}/{}_{}_t{}", &path_network_csv, "sc", &file_name_network_csv, params::TIME-1).as_str());
                    scenario_random_rewiring.export_network_csv(format!("{}/{}_{}_t{}", &path_network_csv, "rr", &file_name_network_csv, params::TIME-1).as_str());
                    scenario_no_rewiring.export_network_csv(format!("{}/{}_{}_t{}", &path_network_csv, "nr", &file_name_network_csv, params::TIME-1).as_str());

                    pb.inc(1); // Increment the progress bar
                });
                pb.finish_with_message("Done!");
        
    }
    
 
}

// Example struct for "Performance" metrics
#[derive(Default)]
struct OutcomeVariable {
    avg: Vec<f64>,
    std: Vec<f64>,
}

// Methods to accumulate values
impl OutcomeVariable {
    fn new() -> Self {
        Self {
            avg: vec![0.0; params::TIME],
            std: vec![0.0; params::TIME],
        }
    }

    // Accumulate a single value at time t
    fn accumulate(&mut self, t: usize, value: f64) {
        self.avg[t] += value;
        self.std[t] += value.powi(2);
    }

    // Finalize the metrics (divide by iteration, compute std)
    fn finalize(&mut self) {
        for (avg, std) in self.avg.iter_mut().zip(&mut self.std) {
            *avg /= params::ITERATION as f64;
            *std /= params::ITERATION as f64;
            *std = (*std - avg.powi(2)).sqrt();
        }
    }
}