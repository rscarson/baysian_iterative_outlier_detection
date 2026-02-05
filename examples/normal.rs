use std::time::Duration;

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Use a normal distribution (mean=0, stddev=1) to generate a dataset of given length.
fn normal_set(len: usize, seed: u64) -> Vec<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    (0..len).map(|_| normal.sample(&mut rng)).collect()
}

/// Our transform will just replace the set with a new normal set.
fn example_transform(data: &[f64], seed: u64) -> Vec<f64> {
    normal_set(data.len(), seed)
}

/// Our test will fail if the magnitude of the sum of the dataset exceeds 3 standard deviations - which is 3
fn example_test(data: &[f64]) {
    /* """ Our test will fail if the magnitude of the sum of the dataset exceeds 3 standard deviations - which is 3 """ */
    const THRESHOLD: f64 = 9.0;
    let total: f64 = data.iter().sum();
    assert!(
        total.abs() < THRESHOLD,
        "Test failed: |sum| = {} >= {}",
        total.abs(),
        THRESHOLD
    );
}

fn main() {
    let options = biod::Options::new(0.95, 0.00001)
        .with_pass_ratio(0.99)
        .with_timeout(Duration::from_secs(60));

    let set = normal_set(1000, 42);
    let runner = biod::Runner::new(&set, options, example_transform, example_test);

    runner.run();
}
