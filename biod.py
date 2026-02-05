import math
import random
import sys
import time

from scipy.stats import beta

DAMPING_CONSTANT = 1.864
PRIOR_STRENGTH = 3.35e6

def p_fail_lower_bound(a, b, q):
    """Calculates the lower bound of the (1-q) confidence interval for a Beta(a, b) distribution."""
    bdist = beta(a, b)
    return bdist.ppf((1.0 - q) / 2.0)

def biod_k(q, p_fail):
    """Calculates the required number of iterations k for BIOD given confidence q and failure probability p_fail."""
    return math.ceil(abs(math.log(1.0 - q) / math.log(1.0 - p_fail)))

def biod_run(data, f_transform, f_test, q, p, pass_ratio, timeout_s = None, C = DAMPING_CONSTANT, p_s = PRIOR_STRENGTH):
    """
    Runs the BIOD algorithm on the given data.

    Parameters:
        - data: The dataset to run BIOD on.
        - f_transform: Function to transform a set = fn(dataset, seed) -> transformed_dataset
        - f_test: Function to test a set = fn(transformed_dataset) -> bool (pass/fail)
        - q: Desired confidence level (e.g., 0.95 for 95% confidence)
        - p: Pointwise failure probability estimate (should underestimate true failure probability)
        - pass_ratio: Required pass ratio to consider the test successful (e.g., 0.99 for 99% pass rate)
        - timeout: Optional timeout for the test function, in seconds (default: None)
        - C: Damping constant (default: 1.864)
        - p_s: Prior strength (default: 3.35e6)
    """
    start_time = time.monotonic()

    # Calculate damping fraction and adjusted setwise failure probability
    n = len(data)
    damping_fraction = 1.0 / (1.0 + (C * n * p))
    p_prime = p * damping_fraction
    p_fail = 1.0 - (1.0 - p_prime)**n

    # Calculate initial k
    k = biod_k(q, p_fail)
    k_initial = k

    # Initialize priors
    alpha = p_s * p_fail
    beta = p_s * (1.0 - p_fail)

    print(
        f"Starting with initial pass count k = {k_initial}, "
        f"confidence = {q * 100:.2f}%, "
        f"required pass ratio = {pass_ratio * 100:.2f}%"
    )

    print(f"\rFinished 0 / {k} iterations. 0 failures reported.", end='')
    sys.stdout.flush()

    iterations = 0
    passes = 0
    unreported_passes = 0
    last_failing_seed = 0
    while iterations < k:
        if iterations - passes > 0 and pass_ratio >= 1.0:
            break

        if timeout_s is not None and (time.monotonic() - start_time) > timeout_s:
            print("\nTimeout reached, terminating BIOD run.")
            break

        print(f"\rFinished {iterations} / {k} iterations. {iterations - passes} failures reported.", end='')
        sys.stdout.flush()

        # Perform transformation and test
        seed = random.getrandbits(64)
        transformed_data = f_transform(data, seed)
        passed = f_test(transformed_data)

        # Record result
        iterations += 1
        if passed:
            passes += 1
            unreported_passes += 1
        elif passes < iterations:
            # Failure detected, update priors and recalculate k
            alpha += 1
            beta += unreported_passes
            unreported_passes = 0

            last_failing_seed = seed
        p_fail = alpha / (alpha + beta)
        k = biod_k(q, p_fail)

    print(f"\rFinished {iterations} / {k} iterations. {iterations - passes} failures reported.\n")

    #
    # Final readout
    #
    failures = iterations - passes
    p_fail_bound = p_fail_lower_bound(failures + 1, passes + 1, q)
    print(
        f"{q * 100:.2f}% confident that the true failure rate among datasets is at least {p_fail_bound * 100:.2f}%.\n"
    )

    ratio = (passes / iterations)
    print(f"{failures}/{iterations} tests failed ({ratio * 100.0:.2f}% pass)")

    if iterations < k:
        print(f"BIOD terminated early after {iterations} iterations - not enough evidence to reach required confidence.")
        return False

    if ratio < pass_ratio:
        print(f"BIOD failed to meet required pass ratio of {pass_ratio * 100:.2f}%.")
        print(f"Observed pass ratio: {ratio * 100:.2f}%.")
        print(f"Last failing seed: {last_failing_seed}.")
        return False

    print("BIOD passed successfully.")
    return True

##
##
## Example run of BIOD
##
##
##

def normal_set(len, seed):
    """ Use a normal distribution (mean=0, stddev=1) to generate a dataset of given length. """
    random.seed(seed)
    return [random.gauss(0, 1) for _ in range(len)]

def example_transform(data, seed):
    """ Our transform will just replace the set with a new normal set. """
    return normal_set(len(data), seed)

def example_test(data):
    """ Our test will fail if the magnitude of the sum of the dataset exceeds 3 standard deviations - which is 3 """
    THRESHOLD = 3.0
    total = sum(data)
    return abs(total) < THRESHOLD

if __name__ == "__main__":
    seed = random.getrandbits(64)
    dataset = normal_set(1000, seed)

    biod_run(
        data=dataset,
        f_transform=example_transform,
        f_test=example_test,
        q=0.95,
        p=0.0000001,
        pass_ratio=0.99,
        timeout_s=60
    )