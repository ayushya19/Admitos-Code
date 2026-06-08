"""
preprocessing.py
================
Radar signal preprocessing utilities for the human activity classification project.

This module isolates the radar pre-processing pipeline from dataset loading,
so the functions can be run, tested, and reused independently.
"""

import numpy as np
from scipy.signal import butter, filtfilt, spectrogram as scipy_spectrogram
from scipy.ndimage import zoom
import config


def parse_dat_file(filepath: str):
    """
    Read a .dat file and return header parameters + raw IQ samples.

    Parameters
    ----------
    filepath : str
        Path to the raw .dat file.

    Returns
    -------
    fc : float
        Carrier frequency (Hz).
    Tsweep : float
        Sweep duration (seconds).
    NTS : int
        Samples per chirp.
    Bw : float
        Chirp bandwidth (Hz).
    IQ : np.ndarray
        1-D complex array of IQ samples.
    """
    with open(filepath, 'r') as fh:
        raw = fh.read().split()

    def to_complex(s: str):
        s = s.strip().replace('i', 'j')
        try:
            return complex(s)
        except ValueError:
            return float(s)

    values = [to_complex(v) for v in raw if v.strip()]
    if len(values) < 5:
        raise ValueError(f"Unexpected .dat file format: {filepath}")

    header = values[:4]
    iq_data = np.array(values[4:], dtype=np.complex128)

    fc = float(header[0].real)
    Tsweep = float(header[1].real) / 1000.0
    NTS = int(header[2].real)
    Bw = float(header[3].real)

    if NTS <= 0:
        raise ValueError(f"Invalid NTS value in file {filepath}: {NTS}")

    return fc, Tsweep, NTS, Bw, iq_data


def oddnumber(n: int) -> int:
    return n if n % 2 == 1 else n - 1


def compute_representations(filepath: str):
    """
    Compute the three radar image representations for one sample.

    Returns
    -------
    spec_img : np.ndarray
        Spectrogram image normalized to [0, 1].
    rt_img : np.ndarray
        Range-time image normalized to [0, 1].
    rd_img : np.ndarray
        Range-Doppler image normalized to [0, 1].
    """
    fc, Tsweep, NTS, Bw, IQ = parse_dat_file(filepath)

    nc = len(IQ) // NTS
    if nc < 4:
        raise ValueError(f"Not enough chirps in file {filepath}: found {nc}")

    fs = NTS / Tsweep
    PRF = 1.0 / Tsweep

    data_time = IQ[:NTS * nc].reshape((NTS, nc), order='F')

    range_fft = np.fft.fftshift(np.fft.fft(data_time, axis=0), axes=0)
    half = NTS // 2
    data_range = range_fft[half:, :]

    n_range = data_range.shape[0] - 1
    if n_range <= 0:
        raise ValueError(f"Range bin extraction failed for {filepath}")

    ns = oddnumber(nc) - 1
    b, a = butter(config.MTI_ORDER, config.MTI_CUTOFF, btype='high')
    data_range_mti = np.zeros((n_range, ns), dtype=np.complex128)
    for k in range(n_range):
        data_range_mti[k, :] = filtfilt(b, a, data_range[k + 1, :ns])

    rt_img = 20 * np.log10(np.abs(data_range_mti) + 1e-30)
    data_rd = np.fft.fftshift(np.fft.fft(data_range_mti, axis=1), axes=1)
    rd_img = 20 * np.log10(np.abs(data_rd) + 1e-30)

    win_len = config.SPEC_WINDOW_LEN
    overlap = int(round(win_len * config.SPEC_OVERLAP_FACTOR))
    nfft = config.SPEC_PAD_FACTOR * win_len
    rbin_low = min(config.RBIN_START - 1, n_range - 1)
    rbin_high = min(config.RBIN_END, n_range)

    spectrogram_sum = None
    for rbin in range(rbin_low, rbin_high):
        _, _, S = scipy_spectrogram(
            data_range_mti[rbin, :],
            fs=PRF,
            window=np.hanning(win_len),
            noverlap=overlap,
            nfft=nfft,
            return_onesided=False,
        )
        S_shifted = np.abs(np.fft.fftshift(S, axes=0))
        spectrogram_sum = S_shifted if spectrogram_sum is None else spectrogram_sum + S_shifted

    if spectrogram_sum is None:
        raise ValueError(f"No spectrogram bins computed for {filepath}")

    spec_img = np.flipud(spectrogram_sum)
    spec_img = 20 * np.log10(spec_img + 1e-30)

    def norm01(x: np.ndarray) -> np.ndarray:
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-30)

    return norm01(spec_img), norm01(rt_img), norm01(rd_img)


def resize_image(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize a 2-D image to (h, w) using bilinear interpolation."""
    if img.ndim != 2:
        raise ValueError("resize_image expects a 2D array")
    zoom_h = h / img.shape[0]
    zoom_w = w / img.shape[1]
    return zoom(img, (zoom_h, zoom_w), order=1)
