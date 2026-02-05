#![allow(clippy::manual_is_multiple_of)]

use rand::{Rng, rngs::ThreadRng};

use crate::{Options, State};

struct PanicHookGuard {
    original_hook: Box<dyn Fn(&std::panic::PanicHookInfo<'_>) + Sync + Send + 'static>,
    payload: std::sync::Arc<std::sync::Mutex<String>>,
}

/// Summary statistics from a completed BIOD run
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    /// Estimated upper bound on the failure probability
    pub p_fail_lower_bound: f64,

    /// Approximate observed ratio of passing tests to total tests
    pub observed_pass_ratio: f64,

    /// Number of observed failures during testing
    pub observed_failures: usize,
}
impl Stats {
    pub fn from_state(state: &State) -> Option<Self> {
        Some(Self {
            p_fail_lower_bound: state.p_fail_lower_bound()?,
            observed_pass_ratio: state.pass_ratio(),
            observed_failures: state.iterations() - state.passes(),
        })
    }
}

/// Current stage of the BIOD process
#[derive(Debug, Clone, PartialEq)]
pub enum Result {
    /// Test is still in progress, and at least one iteration remains
    Ongoing,

    /// Test has timed out before completion
    TimedOut,

    /// Test has completed successfully, meeting the required confidence and pass ratio
    Passed(Stats),

    /// Test has completed unsuccessfully, failing to meet the required confidence or pass ratio
    Failed(Stats),

    /// Test has failed due to an internal error
    Error(String),
}

pub struct Runner<'src, T, FTransform, FTest>
where
    T: std::panic::RefUnwindSafe,
    FTransform: Fn(&[T], u64) -> Vec<T>,
    FTest: Fn(&[T]) + std::panic::RefUnwindSafe,
{
    original_data: &'src [T],
    state: State,

    transform: FTransform,
    test: FTest,

    rng: ThreadRng,
    last_err: Option<(Box<dyn std::any::Any + Send>, u64)>,
}
impl<'src, T, FTransform, FTest> Runner<'src, T, FTransform, FTest>
where
    T: std::panic::RefUnwindSafe,
    FTransform: Fn(&[T], u64) -> Vec<T>,
    FTest: Fn(&[T]) + std::panic::RefUnwindSafe,
{
    pub fn new(data: &'src [T], options: Options, transform: FTransform, test: FTest) -> Self {
        Self {
            original_data: data,
            state: State::new(options, data.len()),
            transform,
            test,

            rng: rand::rng(),
            last_err: None,
        }
    }

    pub fn run(mut self) -> Result {
        let hook = Self::start_panic_capture();
        let state = &self.state;

        println!(
            "Starting with initial pass count k = {}, confidence = {:.2}%, required pass ratio = {:.2}%",
            state.initial_k(),
            state.q() * 100.0,
            state.options().pass_ratio * 100.0
        );

        print!(
            "\rFinished 0 / {} iterations. 0 failures reported.",
            state.current_k(),
        );
        std::io::Write::flush(&mut std::io::stdout()).ok();

        loop {
            let result = self.step();
            match &result {
                Result::Ongoing => {
                    if self.state.current_k() < 1_000 || self.state.iterations() % 10 == 0 {
                        let i = self.state.iterations();
                        let k = self.state.current_k().max(i);
                        let failures = i - self.state.passes();
                        let p_fail = self.state.p_fail() * 100.0;
                        print!(
                            "\rFinished {i} / {k} iterations. {failures} failures reported. Estimated >={p_fail:.2}% of sets induce failure."
                        );
                        std::io::Write::flush(&mut std::io::stdout()).ok();
                    }

                    continue;
                }

                Result::TimedOut => {
                    eprintln!("Test ended due to timeout before sufficient evidence was gathered.");
                }

                Result::Error(msg) => {
                    eprintln!("Internal error: {msg}");
                }

                Result::Passed(stats) | Result::Failed(stats) => {
                    let fail_low = stats.p_fail_lower_bound * 100.0;
                    let confidence = self.state.q() * 100.0;
                    println!(
                        "\n\n{confidence:.2}% confident that the true failure rate among datasets is at least {fail_low:.2}%",
                    );

                    let failures = stats.observed_failures;
                    let total_iters = self.state.iterations();
                    let ratio = stats.observed_pass_ratio * 100.0;
                    eprintln!("{failures}/{total_iters} tests failed ({ratio:.2}% pass)\n");

                    if let Result::Failed(_) = result
                        && let Some((err, seed)) = self.last_err
                    {
                        let payload = hook.payload.lock().unwrap().clone();
                        eprintln!("Test failed.");
                        eprintln!("Last failing seed: 0x{seed:0X}");
                        eprintln!("{payload}");
                        std::panic::resume_unwind(err);
                    }
                }
            }

            Self::stop_panic_capture(hook);
            return self.result();
        }
    }

    fn result(&self) -> Result {
        if self.state.should_stop() {
            let stats = match Stats::from_state(&self.state) {
                Some(s) => s,
                None => {
                    return Result::Error(
                        "Failed to compute lower bound on failure probability.".to_string(),
                    );
                }
            };

            return if self.state.has_passed() {
                Result::Passed(stats)
            } else {
                Result::Failed(stats)
            };
        } else if self.state.has_timed_out() {
            return Result::TimedOut;
        }

        Result::Ongoing
    }

    fn step(&mut self) -> Result {
        if let Result::Ongoing = self.result() {
            let seed: u64 = self.rng.random();
            let transformed_data = (self.transform)(self.original_data, seed);
            let result = std::panic::catch_unwind(|| {
                (self.test)(&transformed_data);
            });

            let passed = result.is_ok();
            self.state.record_result(passed);

            if let Err(err) = result {
                self.last_err = Some((err, seed));
            }
        }

        self.result()
    }

    fn start_panic_capture() -> PanicHookGuard {
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

        PanicHookGuard {
            original_hook: default_hook,
            payload: last_payload,
        }
    }

    fn stop_panic_capture(guard: PanicHookGuard) {
        std::panic::set_hook(guard.original_hook);
    }
}
