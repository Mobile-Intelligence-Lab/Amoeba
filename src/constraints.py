import torch
import numpy as np

# Perturbation bounds
MAX_EPS = 0.2
MAX_EPS_TIME = 0.03

# Monotonically Increasing Features (MIF) indices
# indices for: min value, start/end percentile features, max value
# -1 if not applicable
MIF_PARAMS = [
    {"min": 13, "start": 14, "end": 22, "max": 12},
    {"min": 29, "start": 30, "end": 38, "max": 28},
    {"min": 45, "start": 46, "end": 54, "max": 44},
    {"min": 61, "start": 62, "end": 70, "max": 60},
    {"min": 77, "start": 78, "end": 86, "max": 76},
    {"min": 93, "start": 94, "end": 102, "max": 92},
    {"min": -1, "start": 110, "end": 118, "max": 107},
    {"min": 125, "start": 126, "end": 134, "max": 124},
    {"min": -1, "start": 142, "end": 150, "max": 139},
    {"min": 157, "start": 158, "end": 166, "max": 156}
]

# Indices for features related to kurtosis.
KURTOSIS_INDICES = [10, 26, 42, 58, 74, 90, 108, 122, 140, 154]
KURTOSIS_INDICES = (np.asarray(KURTOSIS_INDICES) - 1).tolist()
# Indices for features related to skewness.
SKEW_INDICES = (np.asarray(KURTOSIS_INDICES) + 1).tolist()

# Ranges of indices for time-related features.
TIME_FEATURE_RANGES = [(23, 38), (55, 70), (87, 102)]
# Flattened list of indices for time-related features.
TIME_INDICES = [
    i for (s, e) in TIME_FEATURE_RANGES for i in range(s - 1, e)
]

SIZE_INDICES = list(set(range(166)) - set(TIME_INDICES) - set(KURTOSIS_INDICES) - set(SKEW_INDICES))
TIME_INDICES = list(set(TIME_INDICES) - set(KURTOSIS_INDICES) - set(SKEW_INDICES))

# Correlated features
CORRELATED_FEATURES = [
    # features computed from other features
    np.asarray([4, 5, 6, 7]),

    # variance = std ** 2
    np.asarray([9, 25, 41, 57, 73, 91, 106, 121, 138, 153])
]
CORRELATED_FEATURES = (np.asarray(CORRELATED_FEATURES, dtype=object) - 1).tolist()


def enforce_mif_constraints(x_adv, mif_params):
    """
    Ensures that the features specified by mif_params are monotonically increasing.
    """

    for mif_param in mif_params:
        min, max, start, end = mif_param["min"], mif_param["max"], mif_param["start"], mif_param["end"]
        min, max, start, end = min - 1, max - 1, start - 1, end - 1

        if min > 0:
            diff_start = torch.clamp(x_adv[:, min] - x_adv[:, start], min=0.)
            x_adv[:, start] = x_adv[:, start] + diff_start

        for idx in range(start, end):
            diff = torch.clamp(x_adv[:, idx] - x_adv[:, idx + 1], min=0.)
            x_adv[:, idx + 1] = x_adv[:, idx + 1] + diff

        if max > 0:
            diff_end = torch.clamp(x_adv[:, end] - x_adv[:, max], min=0.)
            x_adv[:, max] = x_adv[:, max] + diff_end

    return x_adv


def enforce_correlation_constraints(x_adv, correlated_features):
    """
    Adjusts features that are computed from other features.
    """

    if correlated_features is None:
        return x_adv

    mask = torch.ones((x_adv.shape[1])).to(x_adv.device)
    mask[list(correlated_features[0]) + list(correlated_features[1])] = 0

    x_adv = x_adv * mask

    x_adv.data[:, 4] = x_adv.data[:, 1] * x_adv.data[:, 38]  # feature 5 = 2 * 39
    x_adv.data[:, 5] = x_adv.data[:, 2] * x_adv.data[:, 70]  # feature 6 = 3 * 71
    x_adv.data[:, 3] = x_adv.data[:, 4] + x_adv.data[:, 5]  # feature 4 = 5 + 6
    x_adv.data[:, 6] = x_adv.data[:, 3] / x_adv.data[:, 0]  # feature 7 = 4 / 1

    # variance = std ** 2
    variance_feature_indices = correlated_features[1]
    std_feature_indices = (np.asarray(variance_feature_indices) - 1).tolist()
    x_adv.data[:, variance_feature_indices] = x_adv.data[:, std_feature_indices] ** 2

    return x_adv


class Constraints:
    """
    Encapsulates the constraints and provides methods to enforce them on adversarial samples.
    """

    def __init__(self, size_indices=[], time_indices=[], mif_params=[], correlated_features=[],
                 max_eps=None, max_eps_time=None):
        self.size_indices = size_indices
        self.time_indices = time_indices
        self.mif_params = mif_params
        self.correlated_features = correlated_features

        self.max_eps = max_eps
        self.max_eps_time = max_eps_time

    def enforce_delta_constraints(self, delta, min_delta=0):
        """
        Enforces constraints on the perturbation magnitude.
        """
        if len(self.size_indices) > 0:
            delta.data[:, self.size_indices] = torch.clamp(delta.data[:, self.size_indices], min=min_delta)

        if self.max_eps is not None:
            delta = torch.clamp(delta, min=-self.max_eps, max=self.max_eps)

        return delta

    def enforce_adv_constraints(self, x_ori, x_adv, clip_min=None, clip_max=None):
        """
        Enforces constraints on the adversarial sample.
        """
        if len(self.mif_params) > 0:
            x_adv = enforce_mif_constraints(x_adv, self.mif_params)

        if self.max_eps is not None:
            eta = torch.clamp(x_adv - x_ori, min=-self.max_eps, max=self.max_eps)
            x_adv = x_ori + eta

        if self.max_eps_time is not None:
            diff = (x_adv - x_ori)[:, self.time_indices]
            eta = torch.clamp(diff, min=-self.max_eps_time, max=self.max_eps_time)
            x_adv[:, self.time_indices] = x_ori[:, self.time_indices] + eta

        if clip_min is not None and clip_max is not None:
            x_adv = torch.clamp(x_adv, min=clip_min, max=clip_max)

        if len(self.correlated_features) > 0:
            x_adv = enforce_correlation_constraints(x_adv, self.correlated_features)

        return x_adv

    def enforce_rnn_adv_constraints(self, x_ori, x_adv, clip_min=None, clip_max=None):
        if self.max_eps is not None:
            eta = torch.clamp((x_adv - x_ori)[:, :, 0], min=-self.max_eps, max=self.max_eps)
            coeff = torch.ones_like(eta)
            x_adv[:, :, 0] = x_ori[:, :, 0] + coeff * eta

        if self.max_eps_time is not None:
            eta = torch.clamp((x_adv - x_ori)[:, :, 1], min=-self.max_eps_time, max=self.max_eps_time)
            coeff = torch.ones_like(eta)
            x_adv[:, :, 1] = x_ori[:, :, 1] + coeff * eta

        if clip_min is not None and clip_max is not None:
            x_adv = torch.clamp(x_adv, min=clip_min, max=clip_max)

        return x_adv
