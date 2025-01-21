use std::env;
use once_cell::sync::OnceCell;
use once_cell::sync::Lazy;

pub static PRINT_NETWORK_CSV: bool = true;

pub static RUN_ID: &str = "DATT";
pub static RUN_DESC: &str = "Standard";

pub static GET_GRAPH: bool = true;
pub static GET_MAT: bool = true;

pub static ITERATION: usize = 10;
pub static LINK_LEVEL: bool = false;
pub static LIMIT_LEVEL: bool = false;
pub static P_ADDITION: f64 = 0.0;

pub static TIME: usize = 751;

pub static INFORMAL_MAX_NUM: usize = 5;
pub static INFORMAL_INITIAL_PROP: f64 = 0.5;
pub static INFORMAL_REWIRING_PROP: f64 = 0.2;

pub static N: usize = 100;
pub static M_OF_BUNDLE: usize = 20;
pub static M_IN_BUNDLE: usize = 5;

pub static SPAN: [usize; 7] = [2, 3, 4, 5, 6, 7, 8];
pub static LENGTH_SPAN: usize = SPAN.len();

// pub static ENFORCEMENT: [f64; 4] = [0.0, 0.5, 0.8, 1.0];
pub static ENFORCEMENT: [f64; 2] = [0.0, 1.0];
pub static LENGTH_ENFORCEMENT: usize = ENFORCEMENT.len();

pub static TURBULENCE_RATE: [f64; 3] = [0.0, 0.1, 0.1];
pub static TURBULENCE_INTERVAL: [usize; 3] = [TIME, 25, 100];
pub static LENGTH_TURBULENCE: usize = TURBULENCE_RATE.len();

pub static TURNOVER_RATE: [f64; 3] = [0.0, 0.01, 0.1];
pub static LENGTH_TURNOVER: usize = TURNOVER_RATE.len();

pub static P_LEARNING: f64 = 0.2;

//SECOND-ORDER PARAMETERS 
pub static M: usize = M_OF_BUNDLE * M_IN_BUNDLE;
pub static M_N: usize = M * N;
pub static N_DYAD: usize = N * (N - 1) / 2;
pub static N_DYAD_F64: f64 = N_DYAD as f64;
pub static M_N_DYAD: f64 = (M * N_DYAD) as f64;

pub static NUM_SOCIAL_DYNAMICS: usize = 2;

pub static INFORMAL_INITIAL_NUM: usize = ((INFORMAL_MAX_NUM * N) as f64 / 2.0 * INFORMAL_INITIAL_PROP) as usize;
pub static INFORMAL_REWIRING_NUM: usize = (INFORMAL_INITIAL_NUM as f64 * INFORMAL_REWIRING_PROP) as usize;
pub static NUM_ADDITION: usize = (N_DYAD as f64 * P_ADDITION) as usize;

pub static CLOSENESS_CENTRALIZATION_DENOMINATOR: f64 = (N as f64 - 1.0) * (N as f64- 2.0)  / (2.0 * N as f64 - 3.0);
pub static CLUSTERING_COEFFICIENT_RANDOM: f64 = (INFORMAL_INITIAL_NUM + N - 1) as f64 / N_DYAD_F64;
pub static CLUSTERING_COEFFICIENT_RANDOM_NO_SOCIAL_DYNAMICS: f64 = (N - 1) as f64 / N_DYAD_F64;

pub static AVERAGE_PATH_LENGTH_RANDOM: Lazy<f64> = Lazy::new(|| (N as f64).ln() / (CLUSTERING_COEFFICIENT_RANDOM * (N-1) as f64).ln());
pub static AVERAGE_PATH_LENGTH_RANDOM_NO_SOCIAL_DYNAMICS: Lazy<f64> = Lazy::new(|| (N as f64).ln() / (CLUSTERING_COEFFICIENT_RANDOM_NO_SOCIAL_DYNAMICS * (N-1) as f64).ln());

pub static RESULT_SHAPE: [usize; 6] = [
    NUM_SOCIAL_DYNAMICS,
    LENGTH_SPAN as usize,
    LENGTH_ENFORCEMENT as usize,
    LENGTH_TURBULENCE as usize,
    LENGTH_TURNOVER as usize,
    TIME,
];

pub static PARAMS_INDEX_COMBINATIONS: OnceCell<Vec<(usize, usize, usize, usize, usize)>> = OnceCell::new();
pub static PARAMS_INDEX_COMBINATIONS_WITH_TIME: OnceCell<Vec<(usize, usize, usize, usize, usize, usize)>> = OnceCell::new();
pub static PARAM_STRING: Lazy<String> = Lazy::new(|| 
    format!(
        "I{}_T{}_LnkLv{}_LmtLv{}_PAdd{}_DMax{}_r0{}_rt{}_N{}_M({}in{})_S{}_E{}_PTurb{}_ITurb{}_PTurn{}_PLrn{}",
        ITERATION,
        TIME,
        if LINK_LEVEL { "1" } else { "0" },
        if LIMIT_LEVEL { "1" } else { "0" },
        P_ADDITION,
        INFORMAL_MAX_NUM,
        INFORMAL_INITIAL_PROP,
        INFORMAL_REWIRING_PROP,
        N,
        M_IN_BUNDLE,
        M_OF_BUNDLE,
        SPAN.iter().map(|x| x.to_string()).collect::<Vec<String>>().join("&"),
        ENFORCEMENT.iter().map(|x| x.to_string()).collect::<Vec<String>>().join("&"),
        TURBULENCE_RATE.iter().map(|x| x.to_string()).collect::<Vec<String>>().join("&"),
        TURBULENCE_INTERVAL.iter().map(|x| x.to_string()).collect::<Vec<String>>().join("&"),
        TURNOVER_RATE.iter().map(|x| x.to_string()).collect::<Vec<String>>().join("&"),
        P_LEARNING
    ).to_string());

pub static FILE_NAME: Lazy<String> = Lazy::new(|| format!("{}_{}", RUN_ID, *PARAM_STRING));
pub static FILE_PATH: Lazy<String> = Lazy::new(|| env::current_dir().unwrap().to_str().unwrap().to_string());

pub fn initialize_once_cells() {

    let mut combinations = Vec::new();
    let mut combinations_with_time = Vec::new();

    for i_social_dynamics in 0..NUM_SOCIAL_DYNAMICS {
        for i_span in 0..LENGTH_SPAN {
            for i_enforcement in 0..LENGTH_ENFORCEMENT {
                for i_turbulence in 0..LENGTH_TURBULENCE {
                    for i_turnover in 0..LENGTH_TURNOVER {
                        combinations.push((
                            i_social_dynamics,
                            i_span,
                            i_enforcement,
                            i_turbulence,
                            i_turnover,
                        ));
                        for t in 0..TIME {
                            combinations_with_time.push((
                                i_social_dynamics,
                                i_span,
                                i_enforcement,
                                i_turbulence,
                                i_turnover,
                                t,
                            ));
                        }
                    }
                }
            }
        }
    }

    PARAMS_INDEX_COMBINATIONS.set(combinations).unwrap();
    PARAMS_INDEX_COMBINATIONS_WITH_TIME.set(combinations_with_time).unwrap();
}