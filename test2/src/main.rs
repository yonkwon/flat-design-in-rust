pub mod params;
pub mod scenario;
pub mod network_analyzer;
pub mod experiment_manager;
pub mod hdf5_manager;

fn main() {

    params::initialize_once_cells();
    experiment_manager::ExperimentManager::run_experiments();

}
