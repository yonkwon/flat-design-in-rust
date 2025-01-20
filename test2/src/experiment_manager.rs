// For parallel iteration
use rayon::prelude::*; 
use crate::scenario::Scenario;
use crate::params;
use indicatif::{ProgressBar, ProgressStyle};

/// Manages the experiment, including running the experiment and processing results.
/// Modify as needed based on your experiment design.
pub struct ExperimentManager {
}

impl ExperimentManager {

    /// Runs the experiment.
    pub fn run_experiments() {
        // Iterate over each combination in parallel
        let total_steps = params::PARAMS_INDEX_COMBINATIONS.get().unwrap().len() * params::ITERATION;
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

                }

                println!("social_dynamics: {}, span: {}, enforcement: {}, turbulence_rate: {}, turnover_rate: {}, performance_avg: {}",
                    i_social_dynamics, 
                    i_span, 
                    i_enforcement, 
                    i_turbulence, 
                    i_turnover, 
                    scenario.performance_avg
                );
                pb.inc(1); // Increment the progress bar
            }
        });
        pb.finish_with_message("Done!");

    }

    // /// Processes results, e.g., saves to a CSV file or prints to console.
    // fn process_results(scenario: Scenario, indices: (usize, usize, usize, usize, usize)) {
    //     let (i_social_dynamics, i_span, i_enforcement, i_turbulence, i_turnover) = indices;
    //     hdf5_manager::write_results(scenario, i_social_dynamics, i_span, i_enforcement, i_turbulence, i_turnover);
    
    //     for (social_dynamics, span, enforcement, turbulence_rate, turnover_rate, performance_avg) in results {
    //         println!(
    //             "social_dynamics: {}, span: {}, enforcement: {}, turbulence_rate: {}, turnover_rate: {}, performance_avg: {}",
    //             social_dynamics, span, enforcement, turbulence_rate, turnover_rate, performance_avg
    //         );
    //     }
    // }
}
