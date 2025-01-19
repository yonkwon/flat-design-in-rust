pub mod params;
pub mod experiment_manager;
pub mod scenario;
pub mod network_analyzer;

fn main() {

    experiment_manager::ExperimentManager::run_experiments();

}
