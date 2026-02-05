use crate::experiments::export_results;

/// Determines C values for various desired k floors
/// Returns the mean and stdev of C values for the desired k floors 3-10 over q in 0.8-0.99
/// 
/// We do 3-10, because <3 iterations is almost meaningless, and >10 is probably too slow for extremely large sets
/// 
/// 0.8-0.99 is chosen as a reasonable range of pass probabilities - <0.8 is no longer a conservative assumption for many datasets
/// 
/// We can do `C = desired_k_floor / (-ln(1-q))` to see values for different desired k floors
pub fn experiment_1_find_c_range() -> (f64, f64) {
    const FILENAME: &str = "damping_experiment1.csv";
    const HEADINGS: [&str; 3] = ["q", "k_floor", "C"];
    const K_RANGE: (usize, usize) = (3, 10);

    let (mut qi, q_min, q_max, q_step) = (0, 0.8, 0.99, 0.01);
    let mut results = Vec::new();
    println!("Starting experiment 1 - finding C values for various k floors and q values");
    while q_min + (qi as f64) * q_step <= q_max {
        let q = q_min + (qi as f64) * q_step;
        qi += 1;

        for k in K_RANGE.0..=K_RANGE.1 {
            let c = (k as f64) / (-(1.0f64 - q).ln());
            results.push([q, k as f64, c]);
        }
    }

    println!("Done; Exporting results to {FILENAME}");
    export_results(FILENAME, HEADINGS, &results);

    let mean = results.iter().map(|r| r[2]).sum::<f64>() / (results.len() as f64);
    let variance = results.iter().map(|r| (r[2] - mean).powi(2)).sum::<f64>() / (results.len() as f64);
    let std_dev = variance.sqrt();

    println!("C = {mean:.4} +/- {std_dev:.4}");
    (mean, std_dev)
}

/// Generates a table of C candidates for various desired k floors and q values
/// 
/// If we take C_mean +/- C_stdev from experiment_1_find_c_range, we can generate a table of k_floors produced by these C candidates over q in 0.8-0.99
pub fn experiment_2_c_candidates_table(c_mean: f64, c_stdev: f64) {
    const FILENAME : &str = "damping_experiment2.csv";
    const HEADINGS: [&str; 5] = ["C\\q", "0.8", "0.9", "0.95", "0.99"];
    const Q_VALUES: [f64; 4] = [0.8, 0.9, 0.95, 0.99];

    let (c_min, c_max) = (c_mean - c_stdev, c_mean + c_stdev);
    let (mut ci, c_step) = (0, (c_max - c_min) / 10.0);
    let mut results = Vec::new();
    println!("Starting experiment 2 - generating C candidates table");
    while c_min + (ci as f64) * c_step <= c_max {
        let c = c_min + (ci as f64) * c_step;
        ci += 1;

        let mut row = [c, 0.0, 0.0, 0.0, 0.0];
        for i in 0..Q_VALUES.len() {
            let q = Q_VALUES[i];
            let k_floor = c * (-(1.0f64 - q).ln());
            row[i + 1] = k_floor.ceil();
        }
        results.push(row);
    }

    println!("Done; Exporting results to {FILENAME}");
    export_results(FILENAME, HEADINGS, &results);
}