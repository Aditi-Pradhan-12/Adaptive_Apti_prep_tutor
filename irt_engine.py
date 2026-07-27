"""
irt_engine.py

A genuine Item Response Theory implementation (proposal section 5.1),
replacing the earlier "last 5 answers -> easy/medium/hard" threshold
heuristic. Uses the Rasch (1-parameter logistic) model:

    P(correct | theta, b) = 1 / (1 + exp(-(theta - b)))

where theta is the learner's estimated ability and b is the item's
difficulty parameter. Ability is estimated via Bayesian updating on a
discretized theta grid (EAP -- Expected A Posteriori estimation), which
is numerically stable even after only one or two responses (unlike
Maximum Likelihood Estimation, which is undefined when a learner has
answered everything correctly or everything incorrectly so far -- a real
edge case that WILL happen in the first few questions of a session).

Each puzzle type (triangle / grid / series) gets its own independent
ability estimate, since being strong at one doesn't imply the same for
another -- consistent with the per-type design already used elsewhere
in this project (weakness diagnosis, difficulty selection, etc).
"""

import numpy as np

# Discretized theta grid: -3 to +3 covers the practically meaningful range
# of ability in a standard IRT scale (roughly "far below average" to
# "far above average").
THETA_GRID = np.linspace(-3, 3, 61)

# Maps the dataset's categorical difficulty labels to IRT difficulty
# parameters (b). These are fixed, sensible defaults for a 3-tier item
# bank; a larger real-world deployment would calibrate these from actual
# response data instead.
DIFFICULTY_TO_B = {"easy": -1.0, "medium": 0.0, "hard": 1.0}
B_TO_DIFFICULTY = {v: k for k, v in DIFFICULTY_TO_B.items()}


def init_posterior() -> np.ndarray:
    """Prior belief over theta before any responses: standard normal N(0,1)."""
    prior = np.exp(-0.5 * THETA_GRID ** 2)
    return prior / prior.sum()


def _likelihood(theta_grid: np.ndarray, b: float, correct: bool) -> np.ndarray:
    """P(observed response | theta) for every theta on the grid, given item difficulty b."""
    p_correct = 1.0 / (1.0 + np.exp(-(theta_grid - b)))
    return p_correct if correct else (1.0 - p_correct)


def update_posterior(posterior: np.ndarray, b: float, correct: bool) -> np.ndarray:
    """Bayesian update: new_posterior ∝ old_posterior * likelihood(response | theta)."""
    likelihood = _likelihood(THETA_GRID, b, correct)
    new_posterior = posterior * likelihood
    total = new_posterior.sum()
    if total <= 0:
        # Numerical safety net -- should not normally trigger, but avoids
        # a divide-by-zero crash in a pathological edge case.
        return init_posterior()
    return new_posterior / total


def estimate_theta(posterior: np.ndarray) -> float:
    """EAP ability estimate: posterior-weighted mean of the theta grid."""
    return float(np.sum(THETA_GRID * posterior))


def estimate_theta_std(posterior: np.ndarray) -> float:
    """Posterior standard deviation -- a measure of how confident/uncertain the estimate still is."""
    mean = estimate_theta(posterior)
    variance = float(np.sum(posterior * (THETA_GRID - mean) ** 2))
    return variance ** 0.5


def select_difficulty(theta: float) -> str:
    """
    Classic IRT item-selection principle: an item provides the most
    information (narrows the ability estimate fastest) when its difficulty
    is close to the learner's current ability estimate. So we pick the
    difficulty tier whose b-value is nearest to theta.
    """
    closest_b = min(DIFFICULTY_TO_B.values(), key=lambda b: abs(b - theta))
    return B_TO_DIFFICULTY[closest_b]


if __name__ == "__main__":
    # Simulate a learner who is consistently strong (true theta ~ +1.5)
    posterior = init_posterior()
    true_theta = 1.5
    rng = np.random.default_rng(42)

    for i in range(15):
        theta_est = estimate_theta(posterior)
        difficulty = select_difficulty(theta_est)
        b = DIFFICULTY_TO_B[difficulty]
        p_correct = 1 / (1 + np.exp(-(true_theta - b)))
        correct = rng.random() < p_correct
        posterior = update_posterior(posterior, b, correct)
        print(f"Q{i+1}: theta_est={theta_est:+.2f} | selected={difficulty} (b={b}) | "
              f"correct={correct} | std={estimate_theta_std(posterior):.2f}")

    print(f"\nFinal theta estimate: {estimate_theta(posterior):+.2f} (true theta was {true_theta})")
