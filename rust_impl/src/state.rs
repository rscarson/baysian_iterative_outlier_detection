use statrs::distribution::ContinuousCDF;

use crate::Options;

/// State for Bayesian Iterative Outlier Detection (BIOD)
///
/// Keeps track of the current state of the BIOD process, including iteration counts,
/// pass counts, and Bayesian parameters.
#[derive(Debug, Clone, Copy)]
pub struct State {
    /// The size of the data set under test
    n: usize,

    /// Options for the BIOD process
    p_s: f64,
    options: Options,

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
impl State {
    const DAMPING_CONSTANT: f64 = 1.864;
    const PRIOR_STRENGTH: f64 = 4.43e5;

    /// Creates a new BIOD state instance with the given options, prior strength, and data set size.
    ///
    /// Only use this if you want to specify a custom prior strength.
    /// Otherwise, use `new()`.
    #[must_use]
    pub fn with_p_s(options: Options, p_s: f64, n: usize) -> Self {
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
    pub fn new(options: Options, n: usize) -> Self {
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

    /// Returns the options used for this BIOD state.
    #[must_use]
    pub fn options(&self) -> Options {
        self.options
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

    /// Determines whether the BIOD process has timed out based on the configured timeout option.
    #[must_use]
    pub fn has_timed_out(&self) -> bool {
        if let Some(timeout) = self.options.timeout {
            let elapsed = self.start_time.elapsed();
            elapsed >= timeout
        } else {
            false
        }
    }

    /// Determines whether the test end, based on current state and options.
    #[must_use]
    pub fn should_stop(&self) -> bool {
        if self.options.pass_ratio >= 1.0 && self.passes < self.iterations {
            // If pass_ratio is 1.0 or greater and we have a failure, stop immediately
            return true;
        }

        // Otherwise, stop if we have enough evidence
        self.iterations >= self.current_k
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
        let bdist = statrs::distribution::Beta::new(
            (self.iterations - self.passes + 1) as f64,
            (self.passes + 1) as f64,
        )
        .ok()?;
        Some(bdist.inverse_cdf((1.0 - self.q()) / 2.0))
    }

    /// Calculates the Clopper-Pearson confidence interval for p_fail at the current confidence level q.
    ///
    /// Used in experimental analysis and debugging.
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
