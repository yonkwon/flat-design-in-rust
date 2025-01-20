use std::time::{SystemTime, UNIX_EPOCH};
use once_cell::sync::OnceCell;
use std::sync::Mutex;

pub static RUN_ID: &str = "DATT";
pub static RUN_DESC: &str = "Standard";

pub static GET_GRAPH: bool = true;
pub static GET_MAT: bool = true;

pub static LINK_LEVEL: bool = false;
pub static P_ADDITION: f64 = 0.0;
pub static DO_POST_REWIRING: bool = true;
pub static LIMIT_LEVEL: bool = false;

pub static ITERATION: usize = 10_000;
pub static TIME: usize = 501;

pub static INFORMAL_MAX_NUM: usize = 5;
pub static INFORMAL_INITIAL_PROP: f64 = 0.5;
pub static INFORMAL_REWIRING_PROP: f64 = 0.2;

pub static N: usize = 100;
pub static M_OF_BUNDLE: usize = 20;
pub static M_IN_BUNDLE: usize = 5;

pub static SPAN: [usize; 7] = [2, 3, 4, 5, 6, 7, 8];
pub static LENGTH_SPAN: usize = SPAN.len();

pub static ENFORCEMENT: [f64; 4] = [0.0, 0.5, 0.8, 1.0];
pub static LENGTH_ENFORCEMENT: usize = ENFORCEMENT.len();

pub static TURBULENCE_RATE: [f64; 1] = [0.0];
pub static TURBULENCE_INTERVAL: [usize; 1] = [TIME];
pub static LENGTH_TURBULENCE: usize = TURBULENCE_RATE.len();

pub static TURNOVER_RATE: [f64; 1] = [0.0];
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

pub static RESULT_KEY_VALUE: [usize; 6] = [
    NUM_SOCIAL_DYNAMICS,
    LENGTH_SPAN as usize,
    LENGTH_ENFORCEMENT as usize,
    LENGTH_TURBULENCE as usize,
    LENGTH_TURNOVER as usize,
    TIME,
];

pub static FILENAME: OnceCell<Mutex<Option<String>>> = OnceCell::new();
pub static PATH_CSV: OnceCell<Mutex<Option<String>>> = OnceCell::new();

pub static TIC: OnceCell<u128> = OnceCell::new();
pub fn initialize_globals() {
    TIC.set(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis()).unwrap();
    FILENAME.set(Mutex::new(None)).unwrap();
    PATH_CSV.set(Mutex::new(None)).unwrap();
}