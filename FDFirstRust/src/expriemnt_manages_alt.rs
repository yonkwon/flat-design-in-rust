use rayon::prelude::*; 
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::sync::{Arc, Mutex};

pub struct ExperimentManager {
    // Your big collection of Arc<Mutex<ArrayD<f64>>> fields, etc...
    // ...
}

impl ExperimentManager {
    pub fn run_experiments_inverted(&mut self) {
        let combos = params::PARAMS_INDEX_COMBINATIONS.get().unwrap();
        
        // Total number of increments = ITERATION * (length of combos) * TIME
        let total_work = combos.len() * params::ITERATION * params::TIME;
        
        let pb_multi = MultiProgress::new();
        let pb_global = pb_multi.add(ProgressBar::new(total_work as u64));
        pb_global.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.green/red}] {pos}/{len} ({eta_precise})")
                .expect("Failed to set progress bar template")
                .progress_chars("#>-"),
        );
        
        println!("Starting experiments with ITERATION on the outside (parallel).");
        
        // ------------------------------------------------------
        // 1) Parallelize over the ITERATION dimension
        // ------------------------------------------------------
        (0..params::ITERATION).into_par_iter().for_each(|iter_idx| {
            // OPTIONAL: Each parallel iteration can have its own local progress bar
            let pb_local = pb_multi.add(ProgressBar::new((combos.len() * params::TIME) as u64));
            pb_local.set_prefix(format!("Iter #{iter_idx} (Thread {:?})", std::thread::current().id()));
            
            // ------------------------------------------------------
            // 2) Inside each iteration, loop over param combos locally
            // ------------------------------------------------------
            for &(
                i_social_dynamics,
                i_span,
                i_enforcement,
                i_turbulence,
                i_turnover,
            ) in combos {
                // Create local accumulators, if needed
                let mut local_perf = OutcomeVariable::new();
                // ... (repeat for other local_* as needed)

                // Unpack actual parameter values
                let span = params::SPAN[i_span];
                let enforcement = params::ENFORCEMENT[i_enforcement];
                let turbulence_rate = params::TURBULENCE_RATE[i_turbulence];
                let turnover_rate = params::TURNOVER_RATE[i_turnover];
                let turbulence_interval = params::TURBULENCE_INTERVAL[i_turbulence];

                // ------------------------------------------------------
                // 3) Run TIME steps for this combo
                // ------------------------------------------------------
                let mut scenario = Scenario::new(
                    i_social_dynamics,
                    span,
                    enforcement,
                    turbulence_rate,
                    turnover_rate,
                );
                // Possibly clone scenario, etc., as in your original code
                // scenario.do_rewiring(...);

                for t in 0..params::TIME {
                    // ... do scenario.step_forward(), gather data, etc.
                    // local_perf.accumulate(t, scenario.performance_avg);
                    // ...
                    
                    // Update the *global* progress bar
                    pb_global.inc(1);
                }

                // finalize local accumulators
                local_perf.finalize();
                // ...

                // Write local data into the Arc<Mutex<>> arrays
                //   e.g., r_perf_avg.lock().unwrap()[...index...] = local_perf.avg[t];
                // 
                // BUT be careful: multiple iterations will be writing
                // to the same array positions if you do this. Usually,
                // you'd accumulate over iteration and then finalize once
                // all iterations are done. So you may need a design where
                // each iteration writes to a separate slice or merges with
                // atomic ops / locks.
            }
            
            pb_local.finish_and_clear();
        });
        
        // After all iterations finish
        pb_global.finish_with_message("Done with parallel iterations!");
    }
}
