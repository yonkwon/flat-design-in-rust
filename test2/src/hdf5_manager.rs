use hdf5_metno::File;
use hdf5_metno::dataset::Dataset;
use crate::params;

pub struct HDFWriter {
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
    // pub r_perf_avg: Dataset,
    // pub r_perf_std: Dataset,
    // pub r_perf_nr_avg: Dataset,
    // pub r_perf_nr_std: Dataset,
    // pub r_perf_rr_avg: Dataset,
    // pub r_perf_rr_std: Dataset,
    // pub r_perf_12_avg: Dataset,
    // pub r_perf_12_std: Dataset,
    // pub r_perf_23_avg: Dataset,
    // pub r_perf_23_std: Dataset,
    // pub r_perf_13_avg: Dataset,
    // pub r_perf_13_std: Dataset,
    // pub r_clws_avg: Dataset,
    // pub r_clws_std: Dataset,
    // pub r_clws_12_avg: Dataset,
    // pub r_clws_12_std: Dataset,
    // pub r_clws_23_avg: Dataset,
    // pub r_clws_23_std: Dataset,
    // pub r_clws_13_avg: Dataset,
    // pub r_clws_13_std: Dataset,
    // pub r_clws_nr_avg: Dataset,
    // pub r_clws_nr_std: Dataset,
    // pub r_clws_rr_avg: Dataset,
    // pub r_clws_rr_std: Dataset,
    // pub r_cent_avg: Dataset,
    // pub r_cent_std: Dataset,
    // pub r_cent_12_avg: Dataset,
    // pub r_cent_12_std: Dataset,
    // pub r_cent_23_avg: Dataset,
    // pub r_cent_23_std: Dataset,
    // pub r_cent_13_avg: Dataset,
    // pub r_cent_13_std: Dataset,
    // pub r_cent_nr_avg: Dataset,
    // pub r_cent_nr_std: Dataset,
    // pub r_cent_rr_avg: Dataset,
    // pub r_cent_rr_std: Dataset,
    // pub r_effi_avg: Dataset,
    // pub r_effi_std: Dataset,
    // pub r_effi_12_avg: Dataset,
    // pub r_effi_12_std: Dataset,
    // pub r_effi_23_avg: Dataset,
    // pub r_effi_23_std: Dataset,
    // pub r_effi_13_avg: Dataset,
    // pub r_effi_13_std: Dataset,
    // pub r_effi_nr_avg: Dataset,
    // pub r_effi_nr_std: Dataset,
    // pub r_effi_rr_avg: Dataset,
    // pub r_effi_rr_std: Dataset,
    // pub r_sigm_avg: Dataset,
    // pub r_sigm_std: Dataset,
    // pub r_sigm_12_avg: Dataset,
    // pub r_sigm_12_std: Dataset,
    // pub r_sigm_23_avg: Dataset,
    // pub r_sigm_23_std: Dataset,
    // pub r_sigm_13_avg: Dataset,
    // pub r_sigm_13_std: Dataset,
    // pub r_sigm_nr_avg: Dataset,
    // pub r_sigm_nr_std: Dataset,
    // pub r_sigm_rr_avg: Dataset,
    // pub r_sigm_rr_std: Dataset,
    // pub r_omeg_avg: Dataset,
    // pub r_omeg_std: Dataset,
    // pub r_omeg_12_avg: Dataset,
    // pub r_omeg_12_std: Dataset,
    // pub r_omeg_23_avg: Dataset,
    // pub r_omeg_23_std: Dataset,
    // pub r_omeg_13_avg: Dataset,
    // pub r_omeg_13_std: Dataset,
    // pub r_omeg_nr_avg: Dataset,
    // pub r_omeg_nr_std: Dataset,
    // pub r_omeg_rr_avg: Dataset,
    // pub r_omeg_rr_std: Dataset,
    // pub r_spva_avg: Dataset,
    // pub r_spva_std: Dataset,
    // pub r_spva_12_avg: Dataset,
    // pub r_spva_12_std: Dataset,
    // pub r_spva_23_avg: Dataset,
    // pub r_spva_23_std: Dataset,
    // pub r_spva_13_avg: Dataset,
    // pub r_spva_13_std: Dataset,
    // pub r_spva_nr_avg: Dataset,
    // pub r_spva_nr_std: Dataset,
    // pub r_spva_rr_avg: Dataset,
    // pub r_spva_rr_std: Dataset,
    // pub perf_seconds: Dataset,
    }

impl HDFWriter {
    pub fn new() -> Self {
        let hdf5_file = File::create(format!("{}{}.h5", params::FILE_PATH.get().unwrap(), params::FILE_NAME.get().unwrap())).unwrap();
        let para_iteration = hdf5_file.new_dataset_builder().with_data(&[params::ITERATION]).create("para_iteration").unwrap();
        let para_time = hdf5_file.new_dataset_builder().with_data(&[params::TIME]).create("para_time").unwrap();
        let para_p_learning = hdf5_file.new_dataset_builder().with_data(&[params::P_LEARNING]).create("para_p_learning").unwrap();
        let para_l_mech = hdf5_file.new_dataset_builder().with_data(&[params::LINK_LEVEL]).create("para_l_mech").unwrap();
        let para_n = hdf5_file.new_dataset_builder().with_data(&[params::N]).create("para_n").unwrap();
        let para_m = hdf5_file.new_dataset_builder().with_data(&[params::M]).create("para_m").unwrap();
        let para_informal_init_p = hdf5_file.new_dataset_builder().with_data(&[params::INFORMAL_INITIAL_PROP]).create("para_informal_init_p").unwrap();
        let para_informal_init_n = hdf5_file.new_dataset_builder().with_data(&[params::INFORMAL_INITIAL_NUM]).create("para_informal_init_n").unwrap();
        let para_informal_rewi_p = hdf5_file.new_dataset_builder().with_data(&[params::INFORMAL_REWIRING_PROP]).create("para_informal_rewi_p").unwrap();
        let para_informal_rewi_n = hdf5_file.new_dataset_builder().with_data(&[params::INFORMAL_REWIRING_NUM]).create("para_informal_rewi_n").unwrap();
        let para_informal_max = hdf5_file.new_dataset_builder().with_data(&[params::INFORMAL_MAX_NUM]).create("para_informal_max").unwrap();
        let para_m_of_bundle = hdf5_file.new_dataset_builder().with_data(&[params::M_OF_BUNDLE]).create("para_m_of_bundle").unwrap();
        let para_m_in_bundle = hdf5_file.new_dataset_builder().with_data(&[params::M_IN_BUNDLE]).create("para_m_in_bundle").unwrap();
        let para_l_span = hdf5_file.new_dataset_builder().with_data(&[params::LENGTH_SPAN]).create("para_l_span").unwrap();
        let para_v_span = hdf5_file.new_dataset_builder().with_data(&params::SPAN).create("para_v_span").unwrap();
        let para_l_enfo = hdf5_file.new_dataset_builder().with_data(&[params::LENGTH_ENFORCEMENT]).create("para_l_enfo").unwrap();
        let para_v_enfo = hdf5_file.new_dataset_builder().with_data(&params::ENFORCEMENT).create("para_v_enfo").unwrap();
        let para_l_turn = hdf5_file.new_dataset_builder().with_data(&[params::LENGTH_TURNOVER]).create("para_l_turn").unwrap();
        let para_v_turn = hdf5_file.new_dataset_builder().with_data(&params::TURNOVER_RATE).create("para_v_turn").unwrap();
        let para_l_turb = hdf5_file.new_dataset_builder().with_data(&[params::LENGTH_TURBULENCE]).create("para_l_turb").unwrap();
        let para_v_turb_r = hdf5_file.new_dataset_builder().with_data(&params::TURBULENCE_RATE).create("para_v_turb_r").unwrap();
        let para_v_turb_i = hdf5_file.new_dataset_builder().with_data(&params::TURBULENCE_INTERVAL).create("para_v_turb_i").unwrap();
        HDFWriter {
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
        }
    }
}