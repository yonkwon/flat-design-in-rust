use hdf5_metno::File;
use hdf5_metno::dataset::Dataset;
use crate::{experiment_manager_alt::ExperimentManager, params};

pub struct HDF5Manager {
    pub hdf5_file: File,
    pub para_iteration: Dataset,
    pub para_time: Dataset,
    pub para_p_learning: Dataset,
    pub para_l_mech: Dataset,
    pub para_n: Dataset,
    pub para_m: Dataset,
    pub para_informal_init_p: Dataset,
    pub para_informal_init_n: Dataset,
    pub para_informal_rewi_p: Dataset,
    pub para_informal_rewi_n: Dataset,
    pub para_informal_max: Dataset,
    pub para_m_of_bundle: Dataset,
    pub para_m_in_bundle: Dataset,
    pub para_l_span: Dataset,
    pub para_v_span: Dataset,
    pub para_l_enfo: Dataset,
    pub para_v_enfo: Dataset,
    pub para_l_turn: Dataset,
    pub para_v_turn: Dataset,
    pub para_l_turb: Dataset,
    pub para_v_turb_r: Dataset,
    pub para_v_turb_i: Dataset,
    pub r_perf_avg: Dataset,
    pub r_perf_std: Dataset,
    pub r_perf_rr_avg: Dataset,
    pub r_perf_rr_std: Dataset,
    pub r_perf_nr_avg: Dataset,
    pub r_perf_nr_std: Dataset,
    pub r_perf_12_avg: Dataset,
    pub r_perf_12_std: Dataset,
    pub r_perf_23_avg: Dataset,
    pub r_perf_23_std: Dataset,
    pub r_perf_13_avg: Dataset,
    pub r_perf_13_std: Dataset,
    pub r_clws_avg: Dataset,
    pub r_clws_std: Dataset,
    pub r_clws_rr_avg: Dataset,
    pub r_clws_rr_std: Dataset,
    pub r_clws_nr_avg: Dataset,
    pub r_clws_nr_std: Dataset,
    pub r_clws_12_avg: Dataset,
    pub r_clws_12_std: Dataset,
    pub r_clws_23_avg: Dataset,
    pub r_clws_23_std: Dataset,
    pub r_clws_13_avg: Dataset,
    pub r_clws_13_std: Dataset,
    pub r_cent_avg: Dataset,
    pub r_cent_std: Dataset,
    pub r_cent_rr_avg: Dataset,
    pub r_cent_rr_std: Dataset,
    pub r_cent_nr_avg: Dataset,
    pub r_cent_nr_std: Dataset,
    pub r_cent_12_avg: Dataset,
    pub r_cent_12_std: Dataset,
    pub r_cent_23_avg: Dataset,
    pub r_cent_23_std: Dataset,
    pub r_cent_13_avg: Dataset,
    pub r_cent_13_std: Dataset,
    pub r_tria_avg: Dataset,
    pub r_tria_std: Dataset,
    pub r_tria_rr_avg: Dataset,
    pub r_tria_rr_std: Dataset,
    pub r_tria_nr_avg: Dataset,
    pub r_tria_nr_std: Dataset,
    pub r_tria_12_avg: Dataset,
    pub r_tria_12_std: Dataset,
    pub r_tria_23_avg: Dataset,
    pub r_tria_23_std: Dataset,
    pub r_tria_13_avg: Dataset,
    pub r_tria_13_std: Dataset,
    pub r_effi_avg: Dataset,
    pub r_effi_std: Dataset,
    pub r_effi_rr_avg: Dataset,
    pub r_effi_rr_std: Dataset,
    pub r_effi_nr_avg: Dataset,
    pub r_effi_nr_std: Dataset,
    pub r_effi_12_avg: Dataset,
    pub r_effi_12_std: Dataset,
    pub r_effi_23_avg: Dataset,
    pub r_effi_23_std: Dataset,
    pub r_effi_13_avg: Dataset,
    pub r_effi_13_std: Dataset,
    pub r_sigm_avg: Dataset,
    pub r_sigm_std: Dataset,
    pub r_sigm_rr_avg: Dataset,
    pub r_sigm_rr_std: Dataset,
    pub r_sigm_nr_avg: Dataset,
    pub r_sigm_nr_std: Dataset,
    pub r_sigm_12_avg: Dataset,
    pub r_sigm_12_std: Dataset,
    pub r_sigm_23_avg: Dataset,
    pub r_sigm_23_std: Dataset,
    pub r_sigm_13_avg: Dataset,
    pub r_sigm_13_std: Dataset,
    pub r_omeg_avg: Dataset,
    pub r_omeg_std: Dataset,
    pub r_omeg_rr_avg: Dataset,
    pub r_omeg_rr_std: Dataset,
    pub r_omeg_nr_avg: Dataset,
    pub r_omeg_nr_std: Dataset,
    pub r_omeg_12_avg: Dataset,
    pub r_omeg_12_std: Dataset,
    pub r_omeg_23_avg: Dataset,
    pub r_omeg_23_std: Dataset,
    pub r_omeg_13_avg: Dataset,
    pub r_omeg_13_std: Dataset,
    pub r_spva_avg: Dataset,
    pub r_spva_std: Dataset,
    pub r_spva_rr_avg: Dataset,
    pub r_spva_rr_std: Dataset,
    pub r_spva_nr_avg: Dataset,
    pub r_spva_nr_std: Dataset,
    pub r_spva_12_avg: Dataset,
    pub r_spva_12_std: Dataset,
    pub r_spva_23_avg: Dataset,
    pub r_spva_23_std: Dataset,
    pub r_spva_13_avg: Dataset,
    pub r_spva_13_std: Dataset,
    pub perf_seconds: Dataset,
    }

impl HDF5Manager {
    pub fn new(experiment_manager:ExperimentManager, time_performance:u64) -> Self {
        let hdf5_file = File::create(format!("{}{}.h5", *params::FILE_PATH, *params::FILE_NAME).as_str()).unwrap();
        let para_iteration = hdf5_file.new_dataset_builder().with_data(&[params::ITERATION]).create("para_iteration").unwrap();
        let para_time = hdf5_file.new_dataset_builder().with_data(&[params::TIME]).create("para_time").unwrap();
        let para_p_learning = hdf5_file.new_dataset_builder().with_data(&[params::P_LEARNING]).create("para_p_learning").unwrap();
        let para_n = hdf5_file.new_dataset_builder().with_data(&[params::N]).create("para_n").unwrap();
        let para_m = hdf5_file.new_dataset_builder().with_data(&[params::M]).create("para_m").unwrap();
        let para_informal_init_p = hdf5_file.new_dataset_builder().with_data(&[params::INFORMAL_INITIAL_PROP]).create("para_informal_init_p").unwrap();
        let para_informal_init_n = hdf5_file.new_dataset_builder().with_data(&[params::INFORMAL_INITIAL_NUM]).create("para_informal_init_n").unwrap();
        let para_informal_rewi_p = hdf5_file.new_dataset_builder().with_data(&[params::INFORMAL_REWIRING_PROP]).create("para_informal_rewi_p").unwrap();
        let para_informal_rewi_n = hdf5_file.new_dataset_builder().with_data(&[params::INFORMAL_REWIRING_NUM]).create("para_informal_rewi_n").unwrap();
        let para_informal_max = hdf5_file.new_dataset_builder().with_data(&[params::INFORMAL_MAX_NUM]).create("para_informal_max").unwrap();
        let para_m_of_bundle = hdf5_file.new_dataset_builder().with_data(&[params::M_OF_BUNDLE]).create("para_m_of_bundle").unwrap();
        let para_m_in_bundle = hdf5_file.new_dataset_builder().with_data(&[params::M_IN_BUNDLE]).create("para_m_in_bundle").unwrap();
        let para_l_mech = hdf5_file.new_dataset_builder().with_data(&[params::NUM_SOCIAL_DYNAMICS]).create("para_l_mech").unwrap();
        let para_l_span = hdf5_file.new_dataset_builder().with_data(&[params::LENGTH_SPAN]).create("para_l_span").unwrap();
        let para_v_span = hdf5_file.new_dataset_builder().with_data(&params::SPAN).create("para_v_span").unwrap();
        let para_l_enfo = hdf5_file.new_dataset_builder().with_data(&[params::LENGTH_ENFORCEMENT]).create("para_l_enfo").unwrap();
        let para_v_enfo = hdf5_file.new_dataset_builder().with_data(&params::ENFORCEMENT).create("para_v_enfo").unwrap();
        let para_l_turn = hdf5_file.new_dataset_builder().with_data(&[params::LENGTH_TURNOVER]).create("para_l_turn").unwrap();
        let para_v_turn = hdf5_file.new_dataset_builder().with_data(&params::TURNOVER_RATE).create("para_v_turn").unwrap();
        let para_l_turb = hdf5_file.new_dataset_builder().with_data(&[params::LENGTH_TURBULENCE]).create("para_l_turb").unwrap();
        let para_v_turb_r = hdf5_file.new_dataset_builder().with_data(&params::TURBULENCE_RATE).create("para_v_turb_r").unwrap();
        let para_v_turb_i = hdf5_file.new_dataset_builder().with_data(&params::TURBULENCE_INTERVAL).create("para_v_turb_i").unwrap();
        let r_perf_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_perf_avg.lock().unwrap().view()).create("r_perf_avg").unwrap();
        let r_perf_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_perf_std.lock().unwrap().view()).create("r_perf_std").unwrap();
        let r_perf_nr_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_perf_nr_avg.lock().unwrap().view()).create("r_perf_nr_avg").unwrap();
        let r_perf_nr_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_perf_nr_std.lock().unwrap().view()).create("r_perf_nr_std").unwrap();
        let r_perf_rr_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_perf_rr_avg.lock().unwrap().view()).create("r_perf_rr_avg").unwrap();
        let r_perf_rr_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_perf_rr_std.lock().unwrap().view()).create("r_perf_rr_std").unwrap();
        let r_perf_12_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_perf_12_avg.lock().unwrap().view()).create("r_perf_12_avg").unwrap();
        let r_perf_12_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_perf_12_std.lock().unwrap().view()).create("r_perf_12_std").unwrap();
        let r_perf_23_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_perf_23_avg.lock().unwrap().view()).create("r_perf_23_avg").unwrap();
        let r_perf_23_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_perf_23_std.lock().unwrap().view()).create("r_perf_23_std").unwrap();
        let r_perf_13_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_perf_13_avg.lock().unwrap().view()).create("r_perf_13_avg").unwrap();
        let r_perf_13_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_perf_13_std.lock().unwrap().view()).create("r_perf_13_std").unwrap();
        let r_clws_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_clws_avg.lock().unwrap().view()).create("r_clws_avg").unwrap();
        let r_clws_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_clws_std.lock().unwrap().view()).create("r_clws_std").unwrap();
        let r_clws_rr_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_clws_rr_avg.lock().unwrap().view()).create("r_clws_rr_avg").unwrap();
        let r_clws_rr_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_clws_rr_std.lock().unwrap().view()).create("r_clws_rr_std").unwrap();
        let r_clws_nr_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_clws_nr_avg.lock().unwrap().view()).create("r_clws_nr_avg").unwrap();
        let r_clws_nr_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_clws_nr_std.lock().unwrap().view()).create("r_clws_nr_std").unwrap();
        let r_clws_12_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_clws_12_avg.lock().unwrap().view()).create("r_clws_12_avg").unwrap();
        let r_clws_12_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_clws_12_std.lock().unwrap().view()).create("r_clws_12_std").unwrap();
        let r_clws_23_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_clws_23_avg.lock().unwrap().view()).create("r_clws_23_avg").unwrap();
        let r_clws_23_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_clws_23_std.lock().unwrap().view()).create("r_clws_23_std").unwrap();
        let r_clws_13_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_clws_13_avg.lock().unwrap().view()).create("r_clws_13_avg").unwrap();
        let r_clws_13_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_clws_13_std.lock().unwrap().view()).create("r_clws_13_std").unwrap();
        let r_cent_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_cent_avg.lock().unwrap().view()).create("r_cent_avg").unwrap();
        let r_cent_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_cent_std.lock().unwrap().view()).create("r_cent_std").unwrap();
        let r_cent_rr_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_cent_rr_avg.lock().unwrap().view()).create("r_cent_rr_avg").unwrap();
        let r_cent_rr_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_cent_rr_std.lock().unwrap().view()).create("r_cent_rr_std").unwrap();
        let r_cent_nr_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_cent_nr_avg.lock().unwrap().view()).create("r_cent_nr_avg").unwrap();
        let r_cent_nr_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_cent_nr_std.lock().unwrap().view()).create("r_cent_nr_std").unwrap();
        let r_cent_12_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_cent_12_avg.lock().unwrap().view()).create("r_cent_12_avg").unwrap();
        let r_cent_12_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_cent_12_std.lock().unwrap().view()).create("r_cent_12_std").unwrap();
        let r_cent_23_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_cent_23_avg.lock().unwrap().view()).create("r_cent_23_avg").unwrap();
        let r_cent_23_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_cent_23_std.lock().unwrap().view()).create("r_cent_23_std").unwrap();
        let r_cent_13_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_cent_13_avg.lock().unwrap().view()).create("r_cent_13_avg").unwrap();
        let r_cent_13_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_cent_13_std.lock().unwrap().view()).create("r_cent_13_std").unwrap();
        let r_tria_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_tria_avg.lock().unwrap().view()).create("r_tria_avg").unwrap();
        let r_tria_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_tria_std.lock().unwrap().view()).create("r_tria_std").unwrap();
        let r_tria_rr_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_tria_rr_avg.lock().unwrap().view()).create("r_tria_rr_avg").unwrap();
        let r_tria_rr_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_tria_rr_std.lock().unwrap().view()).create("r_tria_rr_std").unwrap();
        let r_tria_nr_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_tria_nr_avg.lock().unwrap().view()).create("r_tria_nr_avg").unwrap();
        let r_tria_nr_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_tria_nr_std.lock().unwrap().view()).create("r_tria_nr_std").unwrap();
        let r_tria_12_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_tria_12_avg.lock().unwrap().view()).create("r_tria_12_avg").unwrap();
        let r_tria_12_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_tria_12_std.lock().unwrap().view()).create("r_tria_12_std").unwrap();
        let r_tria_23_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_tria_23_avg.lock().unwrap().view()).create("r_tria_23_avg").unwrap();
        let r_tria_23_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_tria_23_std.lock().unwrap().view()).create("r_tria_23_std").unwrap();
        let r_tria_13_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_tria_13_avg.lock().unwrap().view()).create("r_tria_13_avg").unwrap();
        let r_tria_13_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_tria_13_std.lock().unwrap().view()).create("r_tria_13_std").unwrap();
        let r_spva_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_spva_avg.lock().unwrap().view()).create("r_spva_avg").unwrap();
        let r_spva_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_spva_std.lock().unwrap().view()).create("r_spva_std").unwrap();
        let r_spva_rr_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_spva_rr_avg.lock().unwrap().view()).create("r_spva_rr_avg").unwrap();
        let r_spva_rr_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_spva_rr_std.lock().unwrap().view()).create("r_spva_rr_std").unwrap();
        let r_spva_nr_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_spva_nr_avg.lock().unwrap().view()).create("r_spva_nr_avg").unwrap();
        let r_spva_nr_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_spva_nr_std.lock().unwrap().view()).create("r_spva_nr_std").unwrap();
        let r_spva_12_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_spva_12_avg.lock().unwrap().view()).create("r_spva_12_avg").unwrap();
        let r_spva_12_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_spva_12_std.lock().unwrap().view()).create("r_spva_12_std").unwrap();
        let r_spva_23_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_spva_23_avg.lock().unwrap().view()).create("r_spva_23_avg").unwrap();
        let r_spva_23_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_spva_23_std.lock().unwrap().view()).create("r_spva_23_std").unwrap();
        let r_spva_13_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_spva_13_avg.lock().unwrap().view()).create("r_spva_13_avg").unwrap();
        let r_spva_13_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_spva_13_std.lock().unwrap().view()).create("r_spva_13_std").unwrap();
        let r_effi_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_effi_avg.lock().unwrap().view()).create("r_effi_avg").unwrap();
        let r_effi_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_effi_std.lock().unwrap().view()).create("r_effi_std").unwrap();
        let r_effi_rr_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_effi_rr_avg.lock().unwrap().view()).create("r_effi_rr_avg").unwrap();
        let r_effi_rr_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_effi_rr_std.lock().unwrap().view()).create("r_effi_rr_std").unwrap();
        let r_effi_nr_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_effi_nr_avg.lock().unwrap().view()).create("r_effi_nr_avg").unwrap();
        let r_effi_nr_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_effi_nr_std.lock().unwrap().view()).create("r_effi_nr_std").unwrap();
        let r_effi_12_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_effi_12_avg.lock().unwrap().view()).create("r_effi_12_avg").unwrap();
        let r_effi_12_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_effi_12_std.lock().unwrap().view()).create("r_effi_12_std").unwrap();
        let r_effi_23_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_effi_23_avg.lock().unwrap().view()).create("r_effi_23_avg").unwrap();
        let r_effi_23_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_effi_23_std.lock().unwrap().view()).create("r_effi_23_std").unwrap();
        let r_effi_13_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_effi_13_avg.lock().unwrap().view()).create("r_effi_13_avg").unwrap();
        let r_effi_13_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_effi_13_std.lock().unwrap().view()).create("r_effi_13_std").unwrap();
        let r_sigm_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_sigm_avg.lock().unwrap().view()).create("r_sigm_avg").unwrap();
        let r_sigm_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_sigm_std.lock().unwrap().view()).create("r_sigm_std").unwrap();
        let r_sigm_rr_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_sigm_rr_avg.lock().unwrap().view()).create("r_sigm_rr_avg").unwrap();
        let r_sigm_rr_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_sigm_rr_std.lock().unwrap().view()).create("r_sigm_rr_std").unwrap();
        let r_sigm_nr_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_sigm_nr_avg.lock().unwrap().view()).create("r_sigm_nr_avg").unwrap();
        let r_sigm_nr_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_sigm_nr_std.lock().unwrap().view()).create("r_sigm_nr_std").unwrap();
        let r_sigm_12_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_sigm_12_avg.lock().unwrap().view()).create("r_sigm_12_avg").unwrap();
        let r_sigm_12_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_sigm_12_std.lock().unwrap().view()).create("r_sigm_12_std").unwrap();
        let r_sigm_23_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_sigm_23_avg.lock().unwrap().view()).create("r_sigm_23_avg").unwrap();
        let r_sigm_23_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_sigm_23_std.lock().unwrap().view()).create("r_sigm_23_std").unwrap();
        let r_sigm_13_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_sigm_13_avg.lock().unwrap().view()).create("r_sigm_13_avg").unwrap();
        let r_sigm_13_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_sigm_13_std.lock().unwrap().view()).create("r_sigm_13_std").unwrap();
        let r_omeg_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_omeg_avg.lock().unwrap().view()).create("r_omeg_avg").unwrap();
        let r_omeg_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_omeg_std.lock().unwrap().view()).create("r_omeg_std").unwrap();
        let r_omeg_rr_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_omeg_rr_avg.lock().unwrap().view()).create("r_omeg_rr_avg").unwrap();
        let r_omeg_rr_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_omeg_rr_std.lock().unwrap().view()).create("r_omeg_rr_std").unwrap();
        let r_omeg_nr_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_omeg_nr_avg.lock().unwrap().view()).create("r_omeg_nr_avg").unwrap();
        let r_omeg_nr_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_omeg_nr_std.lock().unwrap().view()).create("r_omeg_nr_std").unwrap();
        let r_omeg_12_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_omeg_12_avg.lock().unwrap().view()).create("r_omeg_12_avg").unwrap();
        let r_omeg_12_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_omeg_12_std.lock().unwrap().view()).create("r_omeg_12_std").unwrap();
        let r_omeg_23_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_omeg_23_avg.lock().unwrap().view()).create("r_omeg_23_avg").unwrap();
        let r_omeg_23_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_omeg_23_std.lock().unwrap().view()).create("r_omeg_23_std").unwrap();
        let r_omeg_13_avg = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_omeg_13_avg.lock().unwrap().view()).create("r_omeg_13_avg").unwrap();
        let r_omeg_13_std = hdf5_file.new_dataset_builder().with_data(&experiment_manager.r_omeg_13_std.lock().unwrap().view()).create("r_omeg_13_std").unwrap();
        let perf_seconds = hdf5_file.new_dataset_builder().with_data(&[time_performance]).create("perf_seconds").unwrap();
        HDF5Manager {
            hdf5_file,
            para_iteration,
            para_time,
            para_p_learning,
            para_l_mech,
            para_n,
            para_m,
            para_informal_init_p,
            para_informal_init_n,
            para_informal_rewi_p,
            para_informal_rewi_n,
            para_informal_max,
            para_m_of_bundle,
            para_m_in_bundle,
            para_l_span,
            para_v_span,
            para_l_enfo,
            para_v_enfo,
            para_l_turn,
            para_v_turn,
            para_l_turb,
            para_v_turb_r,
            para_v_turb_i,
            r_perf_avg,
            r_perf_std,
            r_perf_rr_avg,
            r_perf_rr_std,
            r_perf_nr_avg,
            r_perf_nr_std,
            r_perf_12_avg,
            r_perf_12_std,
            r_perf_23_avg,
            r_perf_23_std,
            r_perf_13_avg,
            r_perf_13_std,
            r_clws_avg,
            r_clws_std,
            r_clws_rr_avg,
            r_clws_rr_std,
            r_clws_nr_avg,
            r_clws_nr_std,
            r_clws_12_avg,
            r_clws_12_std,
            r_clws_23_avg,
            r_clws_23_std,
            r_clws_13_avg,
            r_clws_13_std,
            r_cent_avg,
            r_cent_std,
            r_cent_rr_avg,
            r_cent_rr_std,
            r_cent_nr_avg,
            r_cent_nr_std,
            r_cent_12_avg,
            r_cent_12_std,
            r_cent_23_avg,
            r_cent_23_std,
            r_cent_13_avg,
            r_cent_13_std,   
            r_tria_avg,
            r_tria_std,
            r_tria_rr_avg,
            r_tria_rr_std,
            r_tria_nr_avg,
            r_tria_nr_std,
            r_tria_12_avg,
            r_tria_12_std,
            r_tria_23_avg,
            r_tria_23_std,
            r_tria_13_avg,
            r_tria_13_std,
            r_spva_avg,
            r_spva_std,
            r_spva_rr_avg,
            r_spva_rr_std,
            r_spva_nr_avg,
            r_spva_nr_std,
            r_spva_12_avg,
            r_spva_12_std,
            r_spva_23_avg,
            r_spva_23_std,
            r_spva_13_avg,
            r_spva_13_std,
            r_effi_avg,
            r_effi_std,
            r_effi_rr_avg,
            r_effi_rr_std,
            r_effi_nr_avg,
            r_effi_nr_std,
            r_effi_12_avg,
            r_effi_12_std,
            r_effi_23_avg,
            r_effi_23_std,
            r_effi_13_avg,
            r_effi_13_std,
            r_sigm_avg,
            r_sigm_std,
            r_sigm_rr_avg,
            r_sigm_rr_std,
            r_sigm_nr_avg,
            r_sigm_nr_std,
            r_sigm_12_avg,
            r_sigm_12_std,
            r_sigm_23_avg,
            r_sigm_23_std,
            r_sigm_13_avg,
            r_sigm_13_std,
            r_omeg_avg,
            r_omeg_std,
            r_omeg_rr_avg,
            r_omeg_rr_std,
            r_omeg_nr_avg,
            r_omeg_nr_std,
            r_omeg_12_avg,
            r_omeg_12_std,
            r_omeg_23_avg,
            r_omeg_23_std,
            r_omeg_13_avg,
            r_omeg_13_std,
            perf_seconds,
        }
    }

    pub fn write_to_file(&self) {
        self.hdf5_file.flush().unwrap();
    }
}