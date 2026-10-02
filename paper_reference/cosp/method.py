"""Paper-aligned numerical components with explicit implementation conventions."""
import numpy as np
from scipy import signal


def prepare_segment(eeg, *, fir_taps):
    """Return imputed, FIR-filtered 4000 x 16 data, or None for dropout.

    fir_taps is deliberately required: the paper specifies cutoff but not order.
    Convention: reject if ANY channel has >10% missing samples, retain exactly
    10%; finite missing values are median-imputed. Use zero-phase filtfilt.
    """
    x = np.asarray(eeg, dtype=float).copy()
    if x.shape != (4000, 16):
        raise ValueError('Expected one 10-second, 400-Hz, 16-channel segment')
    if not isinstance(fir_taps, int) or fir_taps < 3 or fir_taps % 2 != 1:
        raise ValueError('fir_taps must be an odd integer >= 3')
    missing = ~np.isfinite(x)
    if np.any(missing.mean(axis=0) > 0.1):
        return None
    x[missing] = np.nan
    medians = np.nanmedian(x, axis=0)
    x = np.where(missing, medians, x)
    taps = signal.firwin(fir_taps, 170, fs=400)
    return signal.filtfilt(taps, [1.0], x, axis=0)


def coherence(eeg, *, detrend):
    """Return (120,109) MSC and frequencies using the legacy 256-point grid.

    Explicit convention: periodic Hann, 50% overlap, bins 0 <= f < 170 Hz.
    Paper describes 0.5–170 Hz at ~1.6 Hz; this grid starts at DC, as in the
    legacy implementation. Exact historical bin selection needs confirmation.
    Pair order is (0,1),(0,2),(1,2),...; no experimental channel permutation.
    """
    x = np.asarray(eeg)
    if x.shape != (4000, 16) or not np.isfinite(x).all():
        raise ValueError('Expected finite preprocessed data with shape (4000,16)')
    if detrend not in ('constant', 'linear', False):
        raise ValueError('Specify constant, linear or False detrending')
    pairs = []
    for j in range(1, 16):
        for i in range(j):
            f, c = signal.coherence(x[:, i], x[:, j], fs=400,
                                    window='hann', nperseg=256, noverlap=128,
                                    nfft=256, detrend=detrend)
            pairs.append(c[f < 170])
    result = np.asarray(pairs)
    if not np.isfinite(result).all():
        raise ValueError('Undefined coherence (for example, constant channels)')
    return result, f[f < 170]


def chronological_split(seizure_ids, seizure_times):
    """Split supplied eligible lead seizures 60/20/20 by chronology, floor cuts."""
    ids, times = np.asarray(seizure_ids), np.asarray(seizure_times, dtype=float)
    if ids.ndim != 1 or times.shape != ids.shape or len(ids) < 5:
        raise ValueError('Supply at least five paired lead seizure IDs and times')
    if len(np.unique(ids)) != len(ids) or not np.isfinite(times).all():
        raise ValueError('IDs must be unique and times finite')
    if len(np.unique(times)) != len(times):
        raise ValueError('Lead seizure timestamps must be unique')
    ordered = ids[np.argsort(times)]
    return tuple(np.split(ordered, [int(len(ids) * .6), int(len(ids) * .8)]))


def warning_intervals(times, probabilities, *, sop_seconds, threshold):
    """Causal full-window probability mean and retriggerable warning intervals.

    times are prediction availability times, 10 s apart. Gaps reset smoothing.
    Returned intervals are half-open warning-state intervals; they do not apply
    SPH or score seizures. SPH=60 s must be enforced separately in evaluation.
    A score must strictly exceed the threshold. Abutting intervals coalesce.
    """
    t, p = np.asarray(times, dtype=float), np.asarray(probabilities, dtype=float)
    if t.ndim != 1 or p.shape != t.shape or not np.isfinite(t).all() or not np.isfinite(p).all():
        raise ValueError('Supply paired finite times and probabilities')
    if np.any(np.diff(t) <= 0) or np.any((p < 0) | (p > 1)):
        raise ValueError('Times must increase and probabilities must be in [0,1]')
    if not np.isfinite(sop_seconds) or sop_seconds < 10 or sop_seconds % 10:
        raise ValueError('SOP must be a positive multiple of 10 seconds')
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must be in [0,1]')
    n = int(sop_seconds / 10)
    means = np.full(len(t), np.nan)
    intervals = []
    start = 0
    for i in range(len(t)):
        if i and not np.isclose(t[i] - t[i - 1], 10):
            start = i
        if i - start + 1 < n:
            continue
        means[i] = p[i - n + 1:i + 1].mean()
        if means[i] > threshold:
            end = t[i] + sop_seconds
            if intervals and t[i] <= intervals[-1][1]:
                intervals[-1][1] = end
            else:
                intervals.append([t[i], end])
    return means, intervals


def select_validation_threshold(thresholds, validation_score):
    """Maximize validation PP via a caller-supplied validation-only evaluator.

    Evaluator returns (seizure sensitivity, time-in-warning fraction). Never
    pass test data through this callback. Ties select the first supplied grid
    value, making grid order an explicit run parameter.
    """
    grid = list(thresholds)
    if not grid or any(not np.isfinite(t) or not 0 <= t <= 1 for t in grid):
        raise ValueError('Supply a nonempty threshold grid within [0,1]')
    scores = []
    for threshold in grid:
        sensitivity, tiw = validation_score(threshold)
        if not all(np.isfinite(v) and 0 <= v <= 1 for v in (sensitivity, tiw)):
            raise ValueError('Validation sensitivity and TiW must be fractions')
        scores.append(sensitivity * (1 - tiw))
    return grid[int(np.argmax(scores))], scores

