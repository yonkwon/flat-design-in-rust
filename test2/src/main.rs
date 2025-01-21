pub mod params;
pub mod scenario;
pub mod network_analyzer;
pub mod experiment_manager;
pub mod hdf5_manager;

fn main() {
    params::initialize_once_cells();
    let mut experiment_manager = experiment_manager::ExperimentManager::new();
    let tic = std::time::Instant::now().elapsed().as_secs();
    experiment_manager.run_experiments();
    if params::PRINT_NETWORK_CSV {
        experiment_manager.sample_network_csv();
    }
    let toc = std::time::Instant::now().elapsed().as_secs();

    let hdf5_manager = hdf5_manager::HDF5Manager::new(experiment_manager, toc-tic);
    hdf5_manager.write_to_file();
}
