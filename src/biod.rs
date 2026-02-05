//! Bayesian Iterative Outlier Detection (BIOD)
//!
//! When testing a polynomial function, or mechanism that retrieves one, it can be helpful to use random transforms on a data set to catch edge cases
//! however this introduces the probability that any given run of the test will fail to catch such an edge case
//!
//! if we assume the probability of such a failure is caused by individual extreme outliers, then the probability of failure scales with the number of points
//! furthermore we can assume that as the data set grows the probability of failure grows, but with diminishing returns, since complex interactions between outliers
//! can cause their effects to cancel each other out, or otherwise smooth out the extreme behavior. We also assume that all tests are independent and use the same size
//! of dataset n.
//!
//! I propose the following algorithm for running tests in such an environment.
//!
//! given:
//! - `n`, the size of the data set under test
//! - `p`, the probability of any individual point being an extreme outlier, chosen very conservatively
//! - `q`, the desired confidence that we have performed enough iterations to discover a theoretical bug caused by an extreme outlier
//! - `pass_ratio`, the required ratio of data sets tested to pass for the overall procedure to be considered to have passed
//!
//! To reflect the diminishing impact of additional data points on the likelihood of failure due to extreme outliers, we use a damping factor to adjust p based on n.
//! This takes the form of a linear fractional damping function `1 / (1 + Cx)`, where C is a constant chosen to reflect the diminishing impact of additional data points
//! and x is the expected number of extreme outliers in the data set, `n * p`.
//!
//! It is intentionally conservative, and in tests produces values for `k` (the required iterations to achieve confidence `q`) that are greater than observed average
//! iterations-to-failure values of k:
//! ```text
//! n: 100, k_obs_avg: 4.41, k_predicted: 25.00
//! n: 1000, k_obs_avg: 0.50, k_predicted: 9.00
//! n: 10000, k_obs_avg: 0.15, k_predicted: 8.00
//! n: 100000, k_obs_avg: 0.07, k_predicted: 8.00
//! ```
//!
//! For the constant C, we can analyze its effect on the theoretical floor of k as n approaches infinity.
//! Using `C = desired_k_floor / (-ln(1-q))`, we can see values for different desired k floors, given a range of q from 0.9 to 0.999
//!
//! If you define the universe of reasonable floors for k to be between 3 and 10 (as values below 3 risk missing bugs, and values above 10 make the test too slow for extreme n),
//! then for a range of q from 0.9 to 0.9999, pi/2 emerges as an approximate constant for C:
//! ```text
//! C mean: 1.9705, C std_dev: 0.8639 (variance: 0.7464), (~PI/1.59)
//! ```
//!
//! Further, pi/2 seems to occupy a flat region of the C vs `k_floor` curve for reasonable q values, where C can be adjusted slightly without large changes to `k_floor`
//! Future work could explore the relationship between q and C more rigorously to find an optimal mapping.
//!
//!
//! we can then calculate an adjusted value for p, the probability of observing an extreme outlier that can trigger a failure
//! `damping_fraction = 1/(1+ pi/2*np)`
//! `p' = p * damping_fraction`
//!
//! we calculate the probability of a single data set failing by:
//!
//! `p_fail = 1 - (1 - p')^n`
//!
//! we can then get the initial estimate for k, the correct number of iterations to achieve q given p as:
//!
//! `k = ceil( |ln(1 - q) / ln(1 - p_fail)| )`
//!
//! We quantify belief in our initial estimate of p_fail as a linear relationship between pass_ratio and p.
//! In experiments this seems to produce a value that balances minimizing false negatives, and keeping k manageable:
//! `p_s = min(p * (1 - pass_ratio), 1)`
//!
//! we begin the test by calculating values for alpha and beta for the update function:
//! `a = p_fail * p_s, b = (1 - p_fail) * p_s`
//!
//! at most k iterations of the test are performed under these conditions, until the number of iterations has been exhausted and the test ends
//!
//! For each reported failure, we increment alpha, and add the number of passes since the last failure to beta, and perform a Bayesian update on `p_fail`:
//! `a +=1, b += passes_since_last_failure, p_fail = a / (a + b)`
//!
//! If `pass_ratio >= 1`, we can instead stop and fail the test after the first failure reported.
//!
//! An updated value of k is calculated using the method above with the new p:
//! `k = ceil( |ln(1 - q) / ln(1 - p_fail)| )`
//!
//! The test ends when the current number of iterations surpasses the current value of k (either when the initial k iterations complete without finding an issue,
//! or when the updated value of k is below the current total number of iterations), or when they predetermined threshold for time has elapsed
//!
//! If the time limit is reached, and the current number of iterations is less than the initial estimate of k, the test is considered to have failed due to insufficient evidence.
//!
//! at the end, if `n_passes/total_iterations > pass_ratio`, the test has passed
//!
//! # Example
//! ```ignore
//! let biod = BayesianIterativeOutlierDetection::new(
//!     BiodOptions {
//!         confidence: 0.95,
//!         outlier_probability: 1e-4,
//!         pass_ratio: 0.95,
//!         timeout: None,
//!     },
//!     n,
//! );
//!
//! let mut state = BIODState::new(biod);
//! while state.should_continue() {
//!     // Run test iteration here, get result
//!     let test_passed = run_test_iteration();
//!
//!     state.record_result(test_passed);
//! }
//!
//! if state.has_passed() {
//!     println!("Test passed with pass ratio: {}", state.pass_ratio());
//! } else {
//!     println!("Test failed with pass ratio: {}", state.pass_ratio());
//! }
//! ```
//!
#![allow(clippy::manual_is_multiple_of)]

use std::time::Duration;

use rand::Rng;
use statrs::distribution::ContinuousCDF;

/// State for Bayesian Iterative Outlier Detection (BIOD)
///
/// Keeps track of the current state of the BIOD process, including iteration counts,
/// pass counts, and Bayesian parameters.
#[derive(Debug, Clone, Copy)]
pub struct BayesianIterativeOutlierDetection {
    /// The size of the data set under test
    n: usize,

    /// Options for the BIOD process
    p_s: f64,
    options: BiodOptions,

    /// The time at which the BIOD process started. Used for timeout calculations.
    start_time: std::time::Instant,

    //
    // Iteration counts
    initial_k: usize,
    current_k: usize,
    iterations: usize,

    //
    // Results counts
    passes: usize,
    unreported_passes: usize,

    //
    // Bayesian parameters
    a: f64,
    b: f64,
}
impl BayesianIterativeOutlierDetection {
    const DAMPING_CONSTANT: f64 = 1.864;
    const PRIOR_STRENGTH: f64 = 4.43e5;

    pub fn with_p_s(options: BiodOptions, p_s: f64, n: usize) -> Self {
        let mut biod = Self {
            n,
            p_s,
            options,
            start_time: std::time::Instant::now(),

            initial_k: 0,
            current_k: 0,
            iterations: 0,

            passes: 0,
            unreported_passes: 0,

            a: 0.0,
            b: 0.0,
        };

        //  Calculate `p_fail`, the probability of a single data set failing:
        //  `p_fail = 1 - (1 - p')^n`
        //
        //  And use `p_s` to calculate the prior alpha and beta values for the Bayesian update
        let p_prime = biod.p_prime();
        let p_fail = 1.0 - (1.0 - p_prime).powi(biod.n as _);

        // Calculate initial k
        biod.initial_k = biod.k(p_fail);
        biod.current_k = biod.initial_k;

        biod.a = p_fail * biod.p_s();
        biod.b = (1.0 - p_fail) * biod.p_s();
        biod
    }

    /// Creates a new BIOD state instance with the given options and data set size.
    #[must_use]
    pub fn new(options: BiodOptions, n: usize) -> Self {
        Self::with_p_s(options, Self::PRIOR_STRENGTH, n)
    }

    pub fn reset(&mut self) {
        self.start_time = std::time::Instant::now();

        self.iterations = 0;
        self.passes = 0;
        self.unreported_passes = 0;

        let p_fail = self.p_fail();
        self.current_k = self.k(p_fail);
    }

    /// Calculate k, the correct number of iterations to achieve q given p
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn k(&self, p_fail: f64) -> usize {
        ((1.0 - self.q()).ln() / (1.0 - p_fail).ln()).abs().ceil() as usize
    }

    /// The pointwise probability of an extreme outlier causing a failure
    #[must_use]
    pub fn p(&self) -> f64 {
        self.options.outlier_probability
    }

    /// The required confidence (`q`) in the initial number of iterations being above
    #[must_use]
    pub fn q(&self) -> f64 {
        self.options.confidence
    }

    /// The prior-belief strength (`p_s`) in the initial outlier probability
    #[must_use]
    pub fn p_s(&self) -> f64 {
        self.p_s
    }

    /// The alpha parameter of the Beta distribution used in Bayesian updating
    #[must_use]
    pub fn a(&self) -> f64 {
        self.a
    }

    /// The beta parameter of the Beta distribution used in Bayesian updating
    #[must_use]
    pub fn b(&self) -> f64 {
        self.b
    }

    /// The initially estimated required number of iterations to achieve confidence `q`
    #[must_use]
    pub fn initial_k(&self) -> usize {
        self.initial_k
    }

    /// The currently estimated required number of iterations to achieve confidence `q`
    #[must_use]
    pub fn current_k(&self) -> usize {
        self.current_k
    }

    /// The total number of iterations performed so far
    #[must_use]
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// The number of passing iterations performed so far
    #[must_use]
    pub fn passes(&self) -> usize {
        self.passes
    }

    /// To reflect the diminishing impact of additional data points on the likelihood of failure due to extreme outliers, we use a damping factor to adjust p based on n.
    /// This takes the form of a linear fractional damping function `1 / (1 + Cx)`, where C is a constant chosen to reflect the diminishing impact of additional data points
    /// and x is the expected number of extreme outliers in the data set, `n * p`.
    ///
    /// It is intentionally conservative, and in tests produces values for `k` (the required iterations to achieve confidence `q`) that are greater than observed average
    /// iterations-to-failure values of k:
    /// ```text
    /// n: 100, k_obs_avg: 4.41, k_predicted: 25.00
    /// n: 1000, k_obs_avg: 0.50, k_predicted: 9.00
    /// n: 10000, k_obs_avg: 0.15, k_predicted: 8.00
    /// n: 100000, k_obs_avg: 0.07, k_predicted: 8.00
    /// ```
    ///
    /// For the constant C, we can analyze its effect on the theoretical floor of k as n approaches infinity.
    /// Using `C = desired_k_floor / (-ln(1-q))`, we can see values for different desired k floors, given a range of q from 0.9 to 0.999
    ///
    /// If you define the universe of reasonable floors for k to be between 3 and 10 (as values below 3 risk missing bugs, and values above 10 make the test too slow for extreme n),
    /// then for a range of q from 0.9 to 0.9999, pi/2 emerges as an approximate constant for C:
    /// ```text
    /// C mean: 1.9705, C std_dev: 0.8639 (variance: 0.7464), (~PI/1.59)
    /// ```
    ///
    /// Further, pi/2 seems to occupy a flat region of the C vs `k_floor` curve for reasonable q values, where C can be adjusted slightly without large changes to `k_floor`
    /// Future work could explore the relationship between q and C more rigorously to find an optimal mapping.
    #[must_use]
    pub fn damping_fraction(&self) -> f64 {
        1.0 / (1.0 + (Self::DAMPING_CONSTANT * self.n as f64 * self.p()))
    }

    /// Calculate p', the adjusted value for p, the probability of observing an extreme outlier that can trigger a failure
    #[must_use]
    pub fn p_prime(&self) -> f64 {
        let damping_fraction = self.damping_fraction();
        self.p() * damping_fraction
    }

    /// Calculate `p_fail`, the probability of a single data set failing:
    /// `p_fail = a / (a + b)`
    #[must_use]
    pub fn p_fail(&self) -> f64 {
        self.a / (self.a + self.b)
    }

    /// Records the result of a test iteration, updating the internal state accordingly.
    pub fn record_result(&mut self, passed: bool) {
        self.iterations += 1;
        if passed {
            self.passes += 1;
            self.unreported_passes += 1;
        }

        if self.passes < self.iterations {
            self.a += 1.0;
            self.b += self.unreported_passes as f64;
            self.unreported_passes = 0;
        }

        let p_fail = self.p_fail();
        self.current_k = self.k(p_fail);
    }

    /// Calculates the current pass ratio.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    #[must_use]
    pub fn pass_ratio(&self) -> f64 {
        self.passes as f64 / self.iterations as f64
    }

    /// Determines whether the test should continue another iteration.
    #[must_use]
    pub fn should_continue(&self) -> bool {
        if self.options.pass_ratio >= 1.0 && self.passes < self.iterations {
            // If pass_ratio is 1.0 or greater and we have a failure, stop immediately
            return false;
        }

        if let Some(timeout) = self.options.timeout {
            let elapsed = self.start_time.elapsed();
            if elapsed >= timeout {
                return false;
            }
        }

        self.iterations < self.current_k
    }

    /// Determines whether there is enough evidence to fulfill q, and the pass ratio requirement.
    #[must_use]
    pub fn has_passed(&self) -> bool {
        self.pass_ratio() >= self.options.pass_ratio && self.has_sufficient_evidence()
    }

    /// Determines whether there is sufficient evidence to fulfill q.
    #[must_use]
    pub fn has_sufficient_evidence(&self) -> bool {
        let min_iterations = self.initial_k.min(self.current_k);
        self.iterations >= min_iterations
    }

    /// Calculates the bounds for p_fail at the current confidence level q.
    ///
    /// Note that only the lower bound is meaningful for BIOD's purposes.
    #[must_use]
    pub fn p_fail_lower_bound(&self) -> Option<f64> {
        let bdist = statrs::distribution::Beta::new(self.a, self.b).ok()?;
        Some(bdist.inverse_cdf(1.0 - self.q()))
    }

    pub fn clopper_pearson(&self) -> Option<(f64, f64)> {
        let alpha = 1.0 - self.options.confidence;
        let f = self.iterations - self.passes;
        let k = self.iterations as f64;
        let s = self.passes as f64;

        let lower = if f == 0 {
            1.0 - (1.0 - alpha).powf(1.0 / k)
        } else {
            let beta = statrs::distribution::Beta::new(f as f64, s + 1.0).ok()?;
            beta.inverse_cdf(alpha / 2.0)
        };

        let upper = if f == self.iterations {
            1.0
        } else {
            let beta = statrs::distribution::Beta::new(f as f64 + 1.0, s).ok()?;
            beta.inverse_cdf(1.0 - alpha / 2.0)
        };

        Some((lower, upper))
    }
}

/// Configuration for Bayesian Iterative Outlier Detection (BIOD)
///
/// Determines the parameters for the BIOD process, including confidence level, outlier probability,
/// belief strength, pass ratio, and optional timeout.
///
/// # Fields
/// - `confidence`: The required confidence (`q`) in the initial number of iterations being above
/// - `outlier_probability`: The expected probability (`p`) of an outlier able to trigger a failure in any given point
/// - `belief_strength`: The strength of belief (`p_s`) in the initial outlier probability
/// - `pass_ratio`: The required ratio of passing tests to total tests for the overall test to be considered a pass
/// - `timeout`: An optional timeout duration for the entire BIOD process
///
/// Below are tables showing the initial required iterations (`k`) for various dataset sizes (`n`) and outlier probabilities (`p`),
/// For commonly used confidence levels (`q`).
///
/// ```text
/// Table for q = 0.8
/// n\p     1.0e-2  1.0e-3  1.0e-4  1.0e-5  1.0e-6  1.0e-7  1.0e-8
/// 100     5       19      164     1612    16097   160947  1609441
/// 1000    3       5       19      164     1612    16097   160947
/// 10000   3       3       5       19      164     1612    16097
/// 100000  3       3       3       5       19      164     1612
/// 1000000 3       3       3       3       5       19      164
///
/// Table for q = 0.9
/// n\p     1.0e-2  1.0e-3  1.0e-4  1.0e-5  1.0e-6  1.0e-7  1.0e-8
/// 100     6       27      234     2307    23030   230263  2302589
/// 1000    4       6       27      234     2307    23030   230263
/// 10000   4       4       6       27      234     2307    23030
/// 100000  4       4       4       6       27      234     2307
/// 1000000 4       4       4       4       6       27      234
///
/// Table for q = 0.95
/// n\p     1.0e-2  1.0e-3  1.0e-4  1.0e-5  1.0e-6  1.0e-7  1.0e-8
/// 100     8       35      305     3001    29963   299578  2995737
/// 1000    6       8       35      305     3001    29963   299578
/// 10000   5       6       8       35      305     3001    29963
/// 100000  5       5       6       8       35      305     3001
/// 1000000 5       5       5       6       8       35      305
///
/// Table for q = 0.99
/// n\p     1.0e-2  1.0e-3  1.0e-4  1.0e-5  1.0e-6  1.0e-7  1.0e-8
/// 100     12      54      468     4613    46059   460525  4605178
/// 1000    8       12      54      468     4613    46059   460525
/// 10000   8       8       12      54      468     4613    46059
/// 100000  8       8       8       12      54      468     4613
/// 1000000 8       8       8       8       12      54      468
/// ```
#[derive(Debug, Clone, Copy)]
pub struct BiodOptions {
    /// The required confidence (`q`) in the initial number of iterations being above
    /// the true number of iterations needed to catch a bug
    ///
    /// Higher values increase the number of iterations needed.
    /// Reasonable values are between 0.9 and 0.99
    pub confidence: f64,

    /// The expected probability (`p`) of an outlier able to trigger a failure in any given point
    ///
    /// Lower values increase the number of iterations needed.
    /// Reasonable values are between 1e-3 and 1e-6
    pub outlier_probability: f64,

    /// The required ratio of passing tests to total tests for the overall test to be considered a pass
    /// between 0.0 and 1.0
    ///
    /// This has no effect on the number of iterations run, only on the final pass/fail determination.
    /// 1.0 will cause the test to fail immediately on the first failure.
    pub pass_ratio: f64,

    /// An optional timeout duration for the entire BIOD process
    /// If specified, the test will end after this duration even if the required number of iterations has not been reached.
    ///
    /// If the timeout is reached before the required number of iterations, the test will be considered a failure.
    pub timeout: Option<Duration>,
}

/// This function repeatedly runs a test function on randomized versions of a dataset to catch rare edge-case failures,
/// using a Bayesian framework to estimate how many iterations are required for statistical confidence.
///
/// When tests use a random component, there is a risk that rare edge cases may not be exercised in a limited number of runs.
/// This function helps mitigate that risk by applying randomized transforms to the dataset and running the test multiple times,
/// estimating the number of iterations needed to achieve a desired confidence level that any potential bugs will be discovered.
///
/// Here is a table summarizing the key parameters and their typical values - for a complete description of the algorithm, see [`BayesianIterativeOutlierDetection`]:
///
/// Parameters: `q = 0.95`, `p = 1e-6`, `p_s = 100.0`, `pass_ratio = 0.9`  
///
/// | n          | p            | p'           | k initial | k after 1 fail | k after 5 fails |
/// |------------|--------------|--------------|-----------|----------------|----------------|
/// | 1          | 1.000000e-6  | 9.999984e-7  | 2,995,736 | 306            | 67             |
/// | 10         | 1.000000e-6  | 9.999843e-7  | 299,578   | 35             | 11             |
/// | 100        | 1.000000e-6  | 9.998429e-7  | 29,963    | 8              | 6              |
/// | 1,000      | 1.000000e-6  | 9.984317e-7  | 3,001     | 6              | 5              |
/// | 10,000     | 1.000000e-6  | 9.845350e-7  | 305       | 5              | 5              |
/// | 100,000    | 1.000000e-6  | 8.642448e-7  | 35        | 5              | 5              |
/// | 1,000,000  | 1.000000e-6  | 3.889845e-7  | 8         | 5              | 5              |
/// | 10,000,000 | 1.000000e-6  | 5.985170e-8  | 6         | 5              | 5              |
/// | 100,000,000| 1.000000e-6  | 6.325926e-9  | 5         | 5              | 5              |
/// |1,000,000,000| 1.000000e-6 | 6.362147e-10 | 5         | 5              | 5              |
///
/// *Notes:*  
/// - For very small `n`, the required initial iterations (`k_initial`) are huge to achieve the target probability `p`.  
/// - The effect of `p_s = 100` smooths the reduction of `k` after failures, preventing overly aggressive drops for small datasets.  
/// - For moderate-to-large `n`, `k` quickly stabilizes at the minimal practical value (5 in this example).  
/// - It is possible to run the test yourself with different parameters using the [`biod`] module.
///
/// Note that a data set size of at least ~1,000 points is recommended to ensure that the initial iteration count is manageable.
///
/// # Overview
/// - Randomized transforms (`transform(data, seed)`) are applied to the dataset before each test iteration.
/// - Each iteration is independent; failures are assumed to be caused by extreme outliers.
/// - The algorithm estimates the number of iterations (`k`) required to achieve a desired confidence (`q`) that
///   any potential bug caused by an extreme outlier will be discovered.
///
/// # Algorithm (simplified)
/// 1. Compute an adjusted outlier probability `p'` to account for diminishing effects in large datasets:
///    ```text
///    frac_survive = 1 / (1 + π/2 * n * p)
///    p' = p * frac_survive
///    ```
/// 2. Compute the probability of a dataset failing:
///    ```text
///    p_fail = 1 - (1 - p')^n
///    ```
/// 3. Compute initial iterations to achieve confidence `q`:
///    ```text
///    k_initial = ceil( ln(1 - q) / ln(1 - p_fail) )
///    ```
/// 4. Initialize Bayesian parameters:
///    ```text
///    a = p_fail * p_s
///    b = (1 - p_fail) * p_s
///    ```
/// 5. Run iterations until:
///    - The current iteration count exceeds the dynamically updated `k`, or
///    - A time limit is reached.
///
/// # Bayesian updating
/// - On failure:
///   ```text
///   a += 1
///   b += passes_since_last_failure
///   p_fail = a / (a + b)
///   k_updated = ceil( ln(1 - q) / ln(1 - p_fail) )
///   ```
///
/// The test ends when the current iteration count surpasses the updated `k` or the time limit is reached.
///
/// # Pass criteria
/// - If `pass_ratio >= 1.0`, the test fails immediately after the first bug.
/// - Otherwise, the test passes if `(n_passes / total_iterations) > pass_ratio`.
///
/// # Parameters
/// - `data`: Original dataset.
/// - `min_pass_rate`: Minimum pass ratio required for the test to succeed (0.0–1.0).
/// - `timeout`: Maximum allowed test duration.
/// - `transform`: Function to produce a randomized dataset from the original using a seed.
/// - `test`: Function that panics if the dataset fails the assertions.
///
/// # Panics
/// - Panics if too many iterations fail according to `min_pass_rate`.
///
/// # Use case
/// - Useful for validating algorithms sensitive to rare edge cases, such as polynomial fitting, numerical stability tests,
///   or implementations of orthogonal polynomials (Chebyshev vs Legendre) with randomized input datasets.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn assert_randomized<T, TransformFn, TestFn>(
    data: &[T],
    options: BiodOptions,
    transform: TransformFn,
    test: TestFn,
) where
    T: std::panic::RefUnwindSafe,
    TransformFn: Fn(&[T], u64) -> Vec<T>,
    TestFn: Fn(&[T]) + std::panic::RefUnwindSafe,
{
    // Require 95% confidence of catching a failure if it exists
    // Assume an outlier probability of 1 in a million
    // Assume a moderate belief in the initial outlier probability
    let mut state = BayesianIterativeOutlierDetection::new(options, data.len());
    println!(
        "assert_randomized: Starting with initial pass count k = {}, confidence = {:.2}%, required pass ratio = {:.2}%",
        state.initial_k(),
        state.q() * 100.0,
        options.pass_ratio * 100.0
    );

    let default_hook = std::panic::take_hook();
    let last_payload = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let hook_payload = std::sync::Arc::clone(&last_payload);

    std::panic::set_hook(Box::new(move |info| {
        let payload = if let Some(s) = info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            String::new()
        };

        *hook_payload.lock().unwrap() = payload;
    }));

    print!(
        "\rFinished 0 / {} iterations. 0 failures reported.",
        state.current_k(),
    );
    std::io::Write::flush(&mut std::io::stdout()).ok();

    let mut last_err = None;
    let mut last_seed = 0u64;
    let mut seed_rng = rand::rng();
    while state.should_continue() {
        let seed: u64 = seed_rng.random();
        let transformed_data = transform(data, seed);
        let result = std::panic::catch_unwind(|| {
            test(&transformed_data);
        });

        let passed = result.is_ok();
        state.record_result(passed);

        if let Err(err) = result {
            last_err = Some(err);
            last_seed = seed;
        }

        if state.current_k() < 1_000 || state.iterations() % 10 == 0 {
            let i = state.iterations();
            let k = state.current_k().max(i);
            let failures = i - state.passes();
            let p_fail = state.p_fail() * 100.0;
            print!(
                "\rFinished {i} / {k} iterations. {failures} failures reported. Estimated {p_fail:.2}% of sets induce failure."
            );
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }

    // Restore the original hook
    std::panic::set_hook(default_hook);

    if !state.has_passed() {
        let total_iters = state.iterations();
        let failures = total_iters - state.passes();
        let ratio = state.pass_ratio();

        if !state.has_sufficient_evidence() {
            panic!(
                "\n\nassert_randomized: Test ended due to timeout before sufficient evidence was gathered."
            );
        }

        let fail_low = state.p_fail_lower_bound().unwrap_or_else(|| {
            panic!(
                "Failed to compute p_fail lower bound; alpha or beta invalid (a={}, b={})",
                state.a(),
                state.b(),
            )
        });

        let confidence = state.q() * 100.0;
        let fail_low = fail_low * 100.0;

        eprintln!(
            "\n\n{confidence:.2}% confident that the true failure rate among datasets is at least {fail_low:.2}%",
        );

        eprintln!(
            "assert_randomized: {failures}/{total_iters} tests failed ({:.2}% pass)\n",
            ratio * 100.0
        );

        eprintln!("Last seed: 0x{:0X}", last_seed);
        if let Some(err) = last_err {
            let payload = last_payload.lock().unwrap().clone();
            eprintln!("{payload}");
            std::panic::resume_unwind(err);
        }
    }
}
