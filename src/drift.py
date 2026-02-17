import numpy as np
from scipy.stats import ks_2samp

def calculate_psi(expected, actual, bins=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    psi = np.sum(
        (actual_perc - expected_perc) *
        np.log((actual_perc + 1e-6) / (expected_perc + 1e-6))
    )
    return psi


def ks_score(y_true, y_prob):
    fraud = y_prob[y_true == 1]
    normal = y_prob[y_true == 0]
    return ks_2samp(fraud, normal).statistic
