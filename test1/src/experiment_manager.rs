use rayon::prelude::*; // For parallel iteration
use crate::params;
use crate::scenario::Scenario;
use std::sync::Mutex;

pub struct ExperimentManager;

impl ExperimentManager {
    /// Runs the experiment.
    pub fn run_experiments() {
        // Collect all combinations of parameters for the Scenario constructor
        let combinations = Self::generate_combinations();

        // Prepare a thread-safe results collector
        let results = Mutex::new(Vec::new());

        // Iterate over each combination in parallel
        combinations.into_par_iter().for_each(|(social_dynamics, span, enforcement, turbulence_rate, turnover_rate)| {
            for _ in 0..params::ITERATION {
                // Create a new Scenario with the given parameters
                let mut scenario = Scenario::new(
                    social_dynamics,
                    span,
                    enforcement,
                    turbulence_rate,
                    turnover_rate,
                );

                // Run the scenario (this depends on your Scenario implementation)
                scenario.step_forward();

                // Collect results (modify as needed based on your data structure)
                let result = (
                    social_dynamics,
                    span,
                    enforcement,
                    turbulence_rate,
                    turnover_rate,
                    scenario.performance_avg, // Example output
                );

                // Push the result into the shared results vector
                results.lock().unwrap().push(result);
            }
        });

        // Access and process results (e.g., save to file or display)
        let results = results.into_inner().unwrap();
        Self::process_results(results);
    }

    /// Generates all combinations of parameters.
    fn generate_combinations() -> Vec<(usize, usize, f64, f64, f64)> {
        let mut combinations = Vec::new();
        for social_dynamics in 0..params::NUM_SOCIAL_DYNAMICS-1 {
            for &span in &params::SPAN {
                for &enforcement in &params::ENFORCEMENT {
                    for &turbulence_rate in &params::TURBULENCE_RATE {
                        for &turnover_rate in &params::TURNOVER_RATE {
                            combinations.push((social_dynamics, span, enforcement, turbulence_rate, turnover_rate));
                        }
                    }
                }
            }
        }
        combinations
    }

    /// Processes results, e.g., saves to a CSV file or prints to console.
    fn process_results(results: Vec<(usize, usize, f64, f64, f64, f64)>) {
        // Example: Print results to console
        for (social_dynamics, span, enforcement, turbulence_rate, turnover_rate, performance_avg) in results {
            println!(
                "social_dynamics: {}, span: {}, enforcement: {}, turbulence_rate: {}, turnover_rate: {}, performance_avg: {}",
                social_dynamics, span, enforcement, turbulence_rate, turnover_rate, performance_avg
            );
        }
    }
}
