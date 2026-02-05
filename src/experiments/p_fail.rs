use rand_distr::Distribution;

use crate::experiments::export_results;

const ITERATIONS: usize = 10_000;

/// Compare observed p_fail over various n to predicted p_fail from BIOD
///
/// We need to make sure that we underestimate the failure probability p_fail
/// as n increases, since this is critical to the correctness of BIOD.
pub fn experiment_1() {
    const FILENAME: &str = "p_fail_experiment1.csv";
    const HEADINGS: [&str; 3] = ["n", "p_fail_simulated", "p_fail_predicted"];

    // Normal distribution with mean 0 and stddev 1
    // Outliers are points with abs(value) > 3.0
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::rng();

    let (mut n_mul, n_max) = (0.0, 5.0);
    let mut results = Vec::new();
    println!("Starting experiment 1");
    while n_mul <= n_max {
        //
        // Determine n, stepping by powers of 10^0.5
        let n = 10f64.powf(n_mul) as usize;
        n_mul += 0.5;

        print!("\rTesting n = {n}...");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        // Determine true p_fail
        let mut pass_count = 0;
        for _ in 0..ITERATIONS {
            if simulate_set_pass(n, &mut rng, &normal) {
                pass_count += 1;
            }
        }
        let p_fail = 1.0 - (pass_count as f64 / ITERATIONS as f64);

        // Determine predicted p_fail from BIOD
        let biod = biod::State::new(
            biod::Options {
                confidence: 0.99,
                outlier_probability: 0.0027, // Approximate probability of a point being outside 3 std devs in a normal distribution
                pass_ratio: 0.0,
                timeout: None,
            },
            n,
        );

        let p_predicted = biod.p_fail();
        results.push([n as f64, p_fail, p_predicted]);
    }

    println!("Done; Exporting results to {FILENAME}");
    export_results(FILENAME, HEADINGS, &results);
}

/// Compare oberved number of iterations k to failure vs BIOD predicted k to failure
///
/// We need to see if the biod predicted number of iterations k would likely contain
/// a failure, given the observed number of iterations k to failure.
pub fn experiment_2() {
    const FILENAME: &str = "p_fail_experiment2.csv";
    const HEADINGS: [&str; 4] = ["n", "k_obs_mean", "k_obs_stdev", "k_predicted"];

    // Normal distribution with mean 0 and stddev 1
    // Outliers are points with abs(value) > 3.0
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::rng();

    let (mut n_mul, n_max) = (0.0, 5.0);
    let mut results = Vec::new();
    println!("Starting experiment 2");
    while n_mul <= n_max {
        // Determine n, stepping by powers of 10^0.5
        let n = 10f64.powf(n_mul) as usize;
        n_mul += 0.5;

        print!("\rTesting n = {n}...");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        // Gather k observations
        let mut k_obs = Vec::new();
        for _ in 0..ITERATIONS {
            let mut k = 0;
            while simulate_set_pass(n, &mut rng, &normal) {
                k += 1;
            }
            k_obs.push(k as f64);
        }

        // Calculate mean and stdev of observed k
        let k_obs_mean = k_obs.iter().cloned().sum::<f64>() / (k_obs.len() as f64);
        let k_obs_stdev = (k_obs
            .iter()
            .cloned()
            .map(|b| (b - k_obs_mean).powi(2))
            .sum::<f64>()
            / (k_obs.len() as f64))
            .sqrt();

        // Determine predicted k from BIOD
        let biod = biod::State::new(
            biod::Options {
                confidence: 0.99,
                outlier_probability: 0.0027, // Approximate probability of a point being outside 3 std devs in a normal distribution
                pass_ratio: 0.0,
                timeout: None,
            },
            n,
        );
        let k_predicted = biod.k(biod.p_fail());

        results.push([n as f64, k_obs_mean, k_obs_stdev, k_predicted as f64]);
    }

    println!("Done; Exporting results to {FILENAME}");
    export_results(FILENAME, HEADINGS, &results);
}

/// Now we test the predictions for p_fail
/// First for all trials we will save the observed p_fail and get a 95% confidence interval
/// We will also use all tries to run a BIOD test and compare the 95% confidence interval from it's beta distribution
pub fn experiment_3() {
    const HEADINGS: [&str; 7] = [
        "n",
        "p",
        "p_fail_true",
        "p_fail_biod_min_mean",
        "p_fail_biod_min_stdev",
        "p_fail_observed",
        "biod_iterations",
    ];
    const P_VALUES: [f64; 3] = [1e-4, 0.0027, 0.01];

    println!("Starting experiment 3");

    // Normal distribution with mean 0 and stddev 1
    // Outliers are points with abs(value) > 3.0
    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let mut rng = rand::rng();

    let (mut n_mul, n_max) = (1.0, 4.0);
    let mut results = [Vec::new(), Vec::new(), Vec::new()];
    while n_mul <= n_max {
        // Determine n, stepping by powers of 10^0.5
        let n = 10f64.powf(n_mul) as usize;
        n_mul += 0.5;

        println!("Testing n = {n}...");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        // Calculate observed failure rate over all iterations
        let (passes, simulation_results) = get_simulation_results(n, &mut rng, &normal);
        let p_fail = 1.0 - (passes as f64 / ITERATIONS as f64);

        for pi in 0..P_VALUES.len() {
            // Setup BIOD with this p value
            let p = P_VALUES[pi];
            let mut biod = biod::State::new(
                biod::Options {
                    confidence: 0.95,
                    outlier_probability: p,
                    pass_ratio: 0.95,
                    timeout: None,
                },
                n,
            );

            // Skip this p if initial_k is too large for ITERATIONS
            if biod.initial_k() * 2 > ITERATIONS {
                println!(
                    " Skipping p = {p:.6} as initial_k ({}) is too large for ITERATIONS ({})",
                    biod.initial_k(),
                    ITERATIONS
                );
                continue;
            }

            let mut obs_failure_rate = 0.0;
            let mut bound_low = Vec::new();
            // We know the sim has at least double the initial_k iterations
            // So we can take a set of samples and average the results
            // simply use a start position offset by a fraction of initial_k
            const BIOD_SAMPLES: usize = 5;
            let step_size = biod.initial_k() / BIOD_SAMPLES;
            for i in 0..BIOD_SAMPLES {
                biod.reset();
                let start = i * step_size;
                for &result in &simulation_results[start..] {
                    if biod.should_stop() {
                        break;
                    } else {
                        biod.record_result(result);
                    }
                }

                let lower = biod
                    .p_fail_lower_bound()
                    .expect("Failed to get p_fail bounds");
                obs_failure_rate += 1.0 - biod.pass_ratio();
                bound_low.push(lower);
            }

            obs_failure_rate /= BIOD_SAMPLES as f64;

            let bound_low_mean = bound_low.iter().cloned().sum::<f64>() / (bound_low.len() as f64);
            let bound_low_stdev = bound_low
                .iter()
                .map(|b| (*b - bound_low_mean).powi(2))
                .sum::<f64>()
                / (bound_low.len() as f64);

            let iterations = biod.iterations();
            results[pi].push([
                n as f64,
                p,
                p_fail,
                bound_low_mean,
                bound_low_stdev,
                obs_failure_rate,
                iterations as f64,
            ]);
            println!(
                " p = {p:.6}, Simulated p_fail: {p_fail:.5}, BIOD 95% CI: [{bound_low_mean:.5} +/- {bound_low_stdev:.5}], Observed p_fail: {obs_failure_rate:.5}, iters = {iterations}"
            );
        }
    }

    println!("Done; Exporting results");

    export_results("p_fail_experiment3_bigp.csv", HEADINGS, &results[2]);
    export_results("p_fail_experiment3_goodp.csv", HEADINGS, &results[1]);
    export_results("p_fail_experiment3_smallp.csv", HEADINGS, &results[0]);
}

fn get_simulation_results(
    n: usize,
    rng: &mut impl rand::Rng,
    dist: &rand_distr::Normal<f64>,
) -> (usize, Vec<bool>) {
    let mut results = Vec::with_capacity(ITERATIONS);
    let mut pass_count = 0;
    for _ in 0..ITERATIONS {
        let pass = simulate_set_pass(n, rng, dist);
        if pass {
            pass_count += 1;
        }
        results.push(pass);
    }
    (pass_count, results)
}

fn simulate_set_pass(n: usize, rng: &mut impl rand::Rng, dist: &rand_distr::Normal<f64>) -> bool {
    // Stddev for the distribution is 1.0, mean is 0.0
    let outlier_sum: f64 = (0..n)
        .map(|_| dist.sample(rng))
        .filter(|x: &f64| x.abs() > 3.0)
        .sum();

    outlier_sum.abs() < 3.0
}
