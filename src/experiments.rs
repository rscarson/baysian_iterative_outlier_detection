use std::path::Path;

mod prior_strength;
pub fn prior_strength_experiments() {
    prior_strength::get_error_for_2_3();

    prior_strength::experiment_1();
    prior_strength::experiment_2();
    prior_strength::experiment_3();

    let output = std::process::Command::new("python")
        .arg(".\\fit_ps.py")
        .output()
        .expect("Failed to execute fit_ps.py");
    println!("{}", String::from_utf8_lossy(&output.stdout));
    println!("{}", String::from_utf8_lossy(&output.stderr));
}

mod p_fail;
pub fn p_fail_experiments() {
    p_fail::experiment_3();
    p_fail::experiment_1();
    p_fail::experiment_2();
}

mod damping;
pub fn damping_experiments() {
    let (c_mean, c_stdev) = damping::experiment_1_find_c_range();
    damping::experiment_2_c_candidates_table(c_mean, c_stdev);
}

fn export_results<const COLS: usize>(
    filename: impl AsRef<Path>,
    headings: [&str; COLS],
    data: &[[f64; COLS]],
) {
    const DIR: &str = "results";
    std::fs::create_dir_all(DIR).expect("Failed to create results directory");
    let filename = Path::new(DIR).join(filename);

    let mut wtr = csv::Writer::from_path(filename).expect("Failed to create CSV writer");

    wtr.write_record(headings)
        .expect("Failed to write CSV header");

    for row in data {
        let string_row: Vec<String> = row.iter().map(|v| v.to_string()).collect();
        wtr.write_record(&string_row)
            .expect("Failed to write CSV row");
    }

    wtr.flush().expect("Failed to flush CSV writer");
}
