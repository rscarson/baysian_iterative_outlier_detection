use std::time::Duration;

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
pub struct Options {
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
impl Options {
    /// Creates a new `Options` with the specified confidence and outlier probability.
    ///
    /// The pass ratio is set to the confidence value by default, and timeout is set to None.
    pub fn new(confidence: f64, outlier_probability: f64) -> Self {
        Self {
            confidence,
            outlier_probability,
            pass_ratio: confidence,
            timeout: None,
        }
    }

    /// Sets the pass ratio for the `Options`.
    ///
    /// The pass ratio is the required ratio of passing tests to total tests for the overall test to be considered a pass.
    pub fn with_pass_ratio(mut self, pass_ratio: f64) -> Self {
        self.pass_ratio = pass_ratio;
        self
    }

    /// Sets the timeout for the `Options`.
    ///
    /// The timeout is an optional duration for the entire BIOD process.
    /// If specified, the test will end after this duration even if the required number of iterations has not been reached.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
}
