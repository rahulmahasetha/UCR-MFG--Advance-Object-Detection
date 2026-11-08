from typing import List, Callable, Tuple, Dict, Any
import numpy as np


def paired_image_bootstrap_test(
    image_ids: List[str],
    eval_fn_a: Callable[[List[str]], float],
    eval_fn_b: Callable[[List[str]], float],
    num_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42
) -> Dict[str, Any]:
    """Paired image-level bootstrap significance test with empirical 95% confidence intervals."""
    rng = np.random.RandomState(seed)
    n = len(image_ids)
    delta_distribution = []

    base_score_a = eval_fn_a(image_ids)
    base_score_b = eval_fn_b(image_ids)
    observed_diff = base_score_b - base_score_a

    for _ in range(num_resamples):
        resampled_indices = rng.choice(n, size=n, replace=True)
        resampled_ids = [image_ids[i] for i in resampled_indices]

        score_a = eval_fn_a(resampled_ids)
        score_b = eval_fn_b(resampled_ids)
        delta_distribution.append(score_b - score_a)

    deltas = np.sort(np.array(delta_distribution))
    ci_lower = np.percentile(deltas, 100 * (alpha / 2.0))
    ci_upper = np.percentile(deltas, 100 * (1.0 - alpha / 2.0))

    # Two-sided empirical p-value
    p_value = 2.0 * min(np.mean(deltas <= 0.0), np.mean(deltas >= 0.0))

    return {
        "observed_diff": float(observed_diff),
        "ci_95": (float(ci_lower), float(ci_upper)),
        "p_value": float(p_value),
        "statistically_significant": bool(ci_lower > 0.0 or ci_upper < 0.0),
    }


def apply_holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Applies Holm-Bonferroni step-down procedure to control family-wise error rate."""
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    significant = [False] * m

    for rank, idx in enumerate(sorted_indices):
        adjusted_alpha = alpha / (m - rank)
        if p_values[idx] <= adjusted_alpha:
            significant[idx] = True
        else:
            break

    return significant
