# coding=utf-8
import numpy as np
from scipy.stats import entropy
from scipy.stats import wasserstein_distance


def forecastabilty(ts):
    """Forecastability Measure.

    Args:
      ts: time series

    Returns:
      1 - the entropy of the fourier transformation of
            time series / entropy of white noise
    """
    ts = (ts - ts.min()) / (ts.max() - ts.min())
    fourier_ts = np.fft.rfft(ts).real
    fourier_ts = (fourier_ts - fourier_ts.min()) / (fourier_ts.max() - fourier_ts.min())
    fourier_ts /= fourier_ts.sum()
    entropy_ts = entropy(fourier_ts)
    fore_ts = 1 - entropy_ts / (np.log(len(ts)))
    if np.isnan(fore_ts):
        return 0
    return fore_ts


def forecastabilty_moving(ts, window, jump=1):
    """Calculates the forecastability of a moving window.

    Args:
      ts: time series
      window: length of slices
      jump: skipped step when taking subslices

    Returns:
      a list of forecastability measures for all slices.
    """

    # ts = Trend(ts).detrend()
    if len(ts) <= 25:
        return forecastabilty(ts)
    fore_lst = np.array(
        [forecastabilty(ts[i - window: i]) for i in np.arange(window, len(ts), jump)]
    )
    fore_lst = fore_lst[~np.isnan(fore_lst)]  # drop nan
    return fore_lst


class Trend:
    """Trend test."""

    def __init__(self, ts):
        self.ts = ts
        self.train_length = len(ts)
        self.a, self.b = self.find_trend(ts)

    def find_trend(self, insample_data):
        # fit a linear regression y=ax+b on the time series
        x = np.arange(len(insample_data))
        a, b = np.polyfit(x, insample_data, 1)
        return a, b

    def detrend(self):
        # remove trend
        return self.ts - (self.a * np.arange(0, len(self.ts), 1) + self.b)

    def inverse_input(self, insample_data):
        # add trend back to the input part of time series
        return insample_data + (self.a * np.arange(0, len(self.ts), 1) + self.b)

    def inverse_pred(self, outsample_data):
        # add trend back to the predictions
        return outsample_data + (
            self.a
            * np.arange(self.train_length, self.train_length + len(outsample_data), 1)
            + self.b
        )


def seasonality_test(original_ts, ppy):
    """Seasonality test.

    Args:
      original_ts: time series
      ppy: periods per year/frequency

    Returns:
      boolean value: whether the TS is seasonal
    """

    s = acf(original_ts, 1)
    for i in range(2, ppy):
        s = s + (acf(original_ts, i) ** 2)

    limit = 1.645 * (np.sqrt((1 + 2 * s) / len(original_ts)))

    return (abs(acf(original_ts, ppy))) > limit


def acf(ts, k):
    """Autocorrelation function.

    Args:
      ts: time series
      k: lag

    Returns:
      acf value
    """
    m = np.mean(ts)
    s1 = 0
    for i in range(k, len(ts)):
        s1 = s1 + ((ts[i] - m) * (ts[i - k] - m))

    s2 = 0
    for i in range(0, len(ts)):
        s2 = s2 + ((ts[i] - m) ** 2)

    return float(s1 / s2)


# Compute spatial indistinguishability metrics for multivariate time series
def compute_similarity_matrices(data, Tp, Tf):
    """
    Compute similarity matrices for historical and future data

    Args:
    data: Multivariate time series data with shape [T, N, D], where:
          T is time steps, N is number of samples, D is feature dimension
    Tp: Historical window length
    Tf: Future window length

    Returns:
    A_P: Historical data similarity matrix with shape [T, N, N]
    A_F: Future data similarity matrix with shape [T, N, N]
    """
    T, N, D = data.shape
    A_P = np.zeros((T, N, N))
    A_F = np.zeros((T, N, N))

    # For each time step
    for t in range(T - Tp - Tf + 1):
        # For each pair of samples
        for i in range(N):
            for j in range(N):
                # Compute historical data similarity A_P
                X_i_past = data[t: t + Tp, i, :]
                X_j_past = data[t: t + Tp, j, :]

                X_i_past_flat = X_i_past.reshape(-1)
                X_j_past_flat = X_j_past.reshape(-1)

                norm_i_past = np.linalg.norm(X_i_past_flat)
                norm_j_past = np.linalg.norm(X_j_past_flat)

                if norm_i_past > 0 and norm_j_past > 0:
                    A_P[t, i, j] = np.dot(X_i_past_flat, X_j_past_flat) / (norm_i_past * norm_j_past)
                else:
                    A_P[t, i, j] = 0

                # Compute future data similarity A_F
                X_i_future = data[t + Tp: t + Tp + Tf, i, :]
                X_j_future = data[t + Tp: t + Tp + Tf, j, :]

                X_i_future_flat = X_i_future.reshape(-1)
                X_j_future_flat = X_j_future.reshape(-1)

                norm_i_future = np.linalg.norm(X_i_future_flat)
                norm_j_future = np.linalg.norm(X_j_future_flat)

                if norm_i_future > 0 and norm_j_future > 0:
                    A_F[t, i, j] = np.dot(X_i_future_flat, X_j_future_flat) / (norm_i_future * norm_j_future)
                else:
                    A_F[t, i, j] = 0

    return A_P, A_F


def compute_indistinguishability_metrics(A_P, A_F, cu=0.9, cl=0.5):
    """
    Compute spatial indistinguishability metrics r1 and r2

    Args:
    A_P: Historical data similarity matrix with shape [T, N, N]
    A_F: Future data similarity matrix with shape [T, N, N]
    cu: Upper threshold, default 0.9
    cl: Lower threshold, default 0.5

    Returns:
    r1: Ratio of indistinguishable samples to total samples
    r2: Ratio of indistinguishable samples to historically similar samples
    """
    T, N, _ = A_P.shape

    # Calculate number of historically similar samples
    similar_historical = np.sum(A_P > cu)

    # Calculate number of indistinguishable samples
    indistinguishable = np.sum((A_P > cu) & (A_F < cl))

    # Calculate total number of samples
    total_samples = T * N * N

    # Calculate metrics r1 and r2
    r1 = indistinguishable / total_samples
    r2 = indistinguishable / similar_historical if similar_historical > 0 else 0

    return r1, r2


def spatial_indistinguishability(data, Tp, Tf, cu=0.9, cl=0.1):
    """
    Compute spatial indistinguishability metrics for multivariate time series

    Args:
    data: Multivariate time series data with shape [T, N, D], where:
          T is time steps, N is number of samples, D is feature dimension
    Tp: Historical window length
    Tf: Future window length
    cu: Upper threshold, default 0.9
    cl: Lower threshold, default 0.1

    Returns:
    r1: Ratio of indistinguishable samples to total samples
    r2: Ratio of indistinguishable samples to historically similar samples
    A_P: Historical data similarity matrix with shape [T, N, N]
    A_F: Future data similarity matrix with shape [T, N, N]
    """
    # Compute similarity matrices
    A_P, A_F = compute_similarity_matrices(data, Tp, Tf)

    # Compute indistinguishability metrics
    r1, r2 = compute_indistinguishability_metrics(A_P, A_F, cu, cl)

    return r1, r2, A_P, A_F


# Distribution drift analysis
def kl_divergence(p, q):
    """
    Compute KL divergence: KL(p||q)
    Choose KL divergence if computational efficiency is the primary concern

    Args:
    p, q: Probability distributions (need to be normalized)

    Returns:
    KL divergence value
    """
    # Ensure p and q are probability distributions (sum to 1)
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Avoid division by zero error
    q = np.where(q < 1e-10, 1e-10, q)

    # Only compute KL divergence where p>0, avoiding 0*log(0/q) cases
    mask = p > 0
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))


def js_divergence(p, q):
    """
    Compute JS divergence, which is a symmetric version of KL divergence
    Use JS divergence if symmetry and higher computational efficiency are needed

    Args:
    p, q: Probability distributions

    Returns:
    JS divergence value
    """
    # Ensure p and q are probability distributions
    p = p / np.sum(p)
    q = q / np.sum(q)

    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def compute_wasserstein(dist1, dist2):
    """
    Compute Wasserstein distance
    Wasserstein distance may be a better metric if distributions may not overlap at all

    Args:
    dist1, dist2: Data samples

    Returns:
    Wasserstein distance
    """
    return wasserstein_distance(dist1, dist2)


def window_drift_analysis(time_series, window_size, step_size=1, metric='js'):
    """
    Analyze distribution drift of time series using sliding window

    Args:
    time_series: Time series data
    window_size: Window size
    step_size: Window sliding step size
    metric: Distance metric to use ('kl', 'js', 'wasserstein')

    Returns:
    drift_scores: Drift scores of each window compared to reference window
    """
    n = len(time_series)
    drift_scores = []

    # Use the first window as reference distribution
    reference_window = time_series[:window_size]

    for i in range(0, n - window_size, step_size):
        current_window = time_series[i: i + window_size]

        if metric == 'kl':
            # Compute histogram
            hist_ref, _ = np.histogram(reference_window, bins=20, density=True)
            hist_curr, _ = np.histogram(current_window, bins=20, density=True)
            score = kl_divergence(hist_ref, hist_curr)
        elif metric == 'js':
            hist_ref, _ = np.histogram(reference_window, bins=20, density=True)
            hist_curr, _ = np.histogram(current_window, bins=20, density=True)
            score = js_divergence(hist_ref, hist_curr)
        elif metric == 'wasserstein':
            score = compute_wasserstein(reference_window, current_window)

        drift_scores.append((i, score))

    return drift_scores
