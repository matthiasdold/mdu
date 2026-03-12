import mne
import numpy as np


def combine_epochs(epo: mne.BaseEpochs, combine: str) -> np.ndarray:
    """Combine epochs data using specified method.

    Parameters
    ----------
    epo : mne.BaseEpochs
        MNE Epochs object containing the data to combine.
    combine : {'mean', 'gfp'}
        Method for combining epochs across channels.
        - 'mean' : Average across channels
        - 'gfp' : Global Field Power (RMS across channels)

    Returns
    -------
    combined_data : np.ndarray
        Combined epoch data with shape (n_epochs, n_times).

    Raises
    ------
    KeyError
        If `combine` is not 'mean' or 'gfp'.

    Examples
    --------
    >>> import mne
    >>> import numpy as np
    >>> # Create sample epochs
    >>> info = mne.create_info(['Ch1', 'Ch2'], sfreq=100, ch_types='eeg')
    >>> data = np.random.randn(10, 2, 100)  # 10 epochs, 2 channels, 100 timepoints
    >>> epochs = mne.EpochsArray(data, info)
    >>> # Combine using mean
    >>> mean_data = combine_epochs(epochs, 'mean')
    >>> mean_data.shape
    (10, 100)
    >>> # Combine using GFP
    >>> gfp_data = combine_epochs(epochs, 'gfp')
    >>> gfp_data.shape
    (10, 100)
    """
    combine_map = {
        "mean": lambda x: x.get_data().mean(axis=1),
        "gfp": lambda x: np.sqrt((x.get_data() ** 2).mean(axis=1)),
    }
    return combine_map[combine](epo)


def bootstrap(
    arr: np.ndarray,
    ci: list[float, float] = [0.025, 0.975],
    min_max: bool = False,
    nboot: int = 2000,
    rng: np.random.Generator | None = None,
    seed: int = 42,
):
    """Compute bootstrap confidence intervals for array data.

    Bootstrap resampling is used to estimate confidence intervals by randomly
    resampling the input data with replacement and computing statistics on each
    bootstrap sample.

    Parameters
    ----------
    arr : np.ndarray
        Input data array with shape (n_samples, ...). The first axis is
        assumed to be the sample dimension.
    ci : list of float, default=[0.025, 0.975]
        Confidence interval bounds as percentiles in range [0, 1].
        Default [0.025, 0.975] gives a 95% confidence interval.
    min_max : bool, default=False
        If True, return min/max across bootstrap samples instead of
        percentile-based confidence intervals.
    nboot : int, default=2000
        Number of bootstrap iterations to perform.
    rng : np.random.Generator or None, default=None
        Random number generator instance. If None, creates a new generator
        using PCG64 algorithm with the specified seed.
    seed : int, default=42
        Random seed for reproducibility when rng is None.

    Returns
    -------
    ci_bounds : np.ndarray
        Confidence interval bounds with shape (2, ...) where the first row
        is the lower bound and the second row is the upper bound.
    bd : np.ndarray
        Bootstrap distribution with shape (nboot, ...) containing all
        bootstrap sample means.

    Examples
    --------
    >>> import numpy as np
    >>> # Create sample data
    >>> data = np.random.randn(100, 5)  # 100 samples, 5 features
    >>> # Compute 95% CI
    >>> ci_bounds, boot_dist = bootstrap(data, ci=[0.025, 0.975], nboot=1000)
    >>> ci_bounds.shape
    (2, 5)
    >>> boot_dist.shape
    (1000, 5)
    >>> # Compute min/max bounds
    >>> minmax_bounds, _ = bootstrap(data, min_max=True, nboot=1000)
    >>> minmax_bounds.shape
    (2, 5)

    Notes
    -----
    The function uses the PCG64 random number generator for better statistical
    properties compared to MT19937. Each bootstrap sample is created by sampling
    with replacement from the input array along the first axis.
    """
    # confidence intervals
    if rng is None:
        # alternative would be MT19937 (not recommended by numpy)       # noqa

        rng = np.random.Generator(np.random.PCG64(seed))
        # mne uses MT19937 which is a bit faster

    bootstrap_idx = rng.integers(0, arr.shape[0], size=(arr.shape[0], nboot))

    bd = np.asarray([arr[idx].mean(axis=0) for idx in bootstrap_idx.T])

    if min_max:
        bd_min = bd.min(axis=0)
        bd_max = bd.max(axis=0)
        return np.array([bd_min, bd_max]), bd
    else:
        ci_low, ci_up = np.percentile(bd, tuple(np.array(ci) * 100), axis=0)
        return np.array([ci_low, ci_up]), bd
