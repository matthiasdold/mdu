import mne
import polars as pl


def mne_epochs_to_polars(epo: mne.BaseEpochs) -> pl.DataFrame:
    """Convert MNE Epochs object to Polars DataFrame.

    Converts epoched EEG/MEG data from MNE format to a long-format Polars DataFrame.
    Data is automatically scaled from Volts to microvolts (µV). If the epochs object
    contains metadata, it is automatically joined to each epoch's data.

    Parameters
    ----------
    epo : mne.BaseEpochs
        MNE Epochs object (e.g., mne.Epochs, mne.EpochsArray) containing epoched data.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with columns:
        - Channel columns: One column per channel with data in µV
        - time: Time in seconds relative to epoch onset
        - epoch_nr: Index identifying each epoch (0 to n_epochs-1)
        - sample_idx: Global continuous sample index across all epochs
        - Metadata columns: Any columns from epo.metadata (if present)

    Examples
    --------
    >>> import mne
    >>> from mdu.mne.mne2dataframe import mne_epochs_to_polars
    >>> # Create sample epochs
    >>> info = mne.create_info(['Ch1', 'Ch2'], sfreq=100, ch_types='eeg')
    >>> data = np.random.randn(5, 2, 50) * 1e-6  # 5 epochs, 2 channels, 50 times
    >>> epochs = mne.EpochsArray(data, info)
    >>> df = mne_epochs_to_polars(epochs)
    >>> df.shape
    (250, 5)  # 5 epochs * 50 timepoints, with Ch1, Ch2, time, epoch_nr, sample_idx

    See Also
    --------
    mne_raw_to_polars : Convert continuous Raw data to DataFrame
    """
    data = (
        epo.get_data() * 1e6
    )  # scale to uV # type: ignore, TODO: derive the scaling from the channel types
    times = epo.times
    channel_names = epo.ch_names

    df = pl.concat(
        [
            pl.DataFrame(data[i, :, :].T, schema=channel_names).with_columns(
                pl.Series("time", times), pl.lit(i).alias("epoch_nr")
            )
            for i in range(data.shape[0])
        ],  # type: ignore
    )

    # Add continuous sample index after concatenation
    df = df.with_row_index("sample_idx")

    if epo.metadata is not None:
        meta_df = pl.from_pandas(epo.metadata).with_row_index("epoch_nr")  # type: ignore
        df = df.join(meta_df, on="epoch_nr", how="left")

    return df


def mne_raw_to_polars(raw: mne.io.BaseRaw) -> pl.DataFrame:
    """Convert MNE Raw object to Polars DataFrame.

    Converts continuous EEG/MEG data from MNE format to a long-format Polars DataFrame.
    Data is automatically scaled from Volts to microvolts (µV).

    Parameters
    ----------
    raw : mne.io.BaseRaw
        MNE Raw object (e.g., mne.io.Raw, mne.io.RawArray) containing continuous data.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with columns:
        - Channel columns: One column per channel with data in µV
        - time: Time in seconds from recording start
        - sample_idx: Sequential sample index (0 to n_times-1)

    Examples
    --------
    >>> import mne
    >>> from mdu.mne.mne2dataframe import mne_raw_to_polars
    >>> # Create sample raw data
    >>> info = mne.create_info(['Ch1', 'Ch2'], sfreq=100, ch_types='eeg')
    >>> data = np.random.randn(2, 1000) * 1e-6  # 2 channels, 1000 timepoints
    >>> raw = mne.io.RawArray(data, info)
    >>> df = mne_raw_to_polars(raw)
    >>> df.shape
    (1000, 4)  # 1000 timepoints, with Ch1, Ch2, time, sample_idx

    See Also
    --------
    mne_epochs_to_polars : Convert epoched data to DataFrame
    """
    data = raw.get_data() * 1e6  # scale to uV # type: ignore
    times = raw.times
    channel_names = raw.ch_names
    df = pl.DataFrame(data.T, schema=channel_names)  # type: ignore
    df = df.with_columns(pl.Series("time", times)).with_row_index("sample_idx")
    return df
