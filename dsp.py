"""Vlastní implementace základních operací zpracování signálu.

Žádná z funkcí nepoužívá nástroje z balíku WFDB (ten slouží jen k načtení dat).
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal as sps
from scipy.fft import dct, irfft, rfft, rfftfreq


# --------------------------------------------------------------------------
# Základní úpravy
# --------------------------------------------------------------------------
def remove_dc(x: np.ndarray) -> np.ndarray:
    """Centralizace signálu kolem podélné osy (odečtení střední hodnoty)."""
    return x - np.mean(x)


def zscore(x: np.ndarray) -> np.ndarray:
    """Nulová střední hodnota a jednotkový rozptyl."""
    x = x - np.mean(x)
    std = np.std(x)
    return x / std if std > 0 else x


def interpolate_nans(x: np.ndarray) -> np.ndarray:
    """Lineární interpolace chybějících (NaN/inf) vzorků."""
    x = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(x)
    if bad.all():
        raise ValueError("Signál neobsahuje žádný platný vzorek.")
    if bad.any():
        idx = np.arange(x.size)
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Převzorkování polyfázovým filtrem (s antialiasingem)."""
    if np.isclose(fs_in, fs_out):
        return np.asarray(x, dtype=float)
    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    return sps.resample_poly(x, ratio.numerator, ratio.denominator)


def moving_average(x: np.ndarray, n: int) -> np.ndarray:
    """Klouzavý průměr o délce okna n vzorků (výstup stejné délky)."""
    n = max(int(n), 1)
    return np.convolve(x, np.ones(n) / n, mode="same")


def bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 2) -> np.ndarray:
    """Nulově fázová pásmová propust (Butterworth, filtfilt)."""
    nyq = fs / 2
    high = min(high, 0.99 * nyq)
    sos = sps.butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sps.sosfiltfilt(sos, x)


def lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    """Nulově fázová dolní propust."""
    sos = sps.butter(order, min(cutoff, 0.99 * fs / 2) / (fs / 2), btype="low", output="sos")
    return sps.sosfiltfilt(sos, x)


def frame_signal(x: np.ndarray, size: int, hop: int) -> np.ndarray:
    """Rozdělení signálu na překrývající se rámce, tvar (n_rámců, size)."""
    if len(x) < size:
        return np.empty((0, size))
    n = 1 + (len(x) - size) // hop
    idx = np.arange(size)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


# --------------------------------------------------------------------------
# Frekvenční oblast, korelace, kepstrum
# --------------------------------------------------------------------------
def magnitude_spectrum(x: np.ndarray, fs: float, nfft: int | None = None):
    """Amplitudové spektrum (Hannovo okno) -> (frekvence, |X|)."""
    nfft = nfft or len(x)
    spec = np.abs(rfft(x * np.hanning(len(x)), nfft))
    return rfftfreq(nfft, 1 / fs), spec


def mean_power_spectrum(x: np.ndarray, fs: float, size: int = 512, hop: int = 256):
    """Průměrné výkonové spektrum přes rámce (vlastní Welchova metoda)."""
    frames = frame_signal(x, size, hop)
    if len(frames) == 0:
        raise ValueError("Signál je kratší než jeden rámec.")
    win = np.hanning(size)
    power = np.abs(rfft(frames * win, size, axis=1)) ** 2
    return rfftfreq(size, 1 / fs), power.mean(axis=0)


def autocorrelation(x: np.ndarray) -> np.ndarray:
    """Normovaná autokorelace (r[0] = 1) pomocí FFT, kladná zpoždění."""
    n = len(x)
    spec = rfft(x, 2 * n)
    r = irfft(np.abs(spec) ** 2)[:n]
    return r / r[0] if r[0] > 0 else r


def cross_correlation_max(a: np.ndarray, b: np.ndarray, max_lag: int):
    """Normovaná vzájemná korelace pro zpoždění v intervalu <-max_lag, max_lag>.

    Vrací (r0, r_max, lag_max): Pearsonův koeficient bez zpoždění, extrém
    s největší absolutní hodnotou a jemu odpovídající zpoždění ve vzorcích
    (kladné = signál b je zpožděn za a).
    """
    n = min(len(a), len(b))
    a, b = zscore(a[:n]), zscore(b[:n])
    c = irfft(rfft(a, 2 * n).conj() * rfft(b, 2 * n))[: 2 * n] / n
    max_lag = min(max_lag, n - 1)
    lags = np.arange(-max_lag, max_lag + 1)
    values = np.concatenate([c[-max_lag:], c[: max_lag + 1]]) if max_lag > 0 else c[:1]
    k = int(np.argmax(np.abs(values)))
    return float(c[0]), float(values[k]), int(lags[k])


def real_cepstrum(frame: np.ndarray, fs: float, nfft: int = 1024):
    """Reálné kepstrum rámce z log. amplitudového spektra v dB.

    Vrací (quefrency [s], kepstrum) pro polovinu nfft.
    """
    spec = np.abs(rfft(frame * np.hanning(len(frame)), nfft))
    ceps = irfft(20 * np.log10(spec + 1e-12), nfft)
    half = nfft // 2
    return np.arange(half) / fs, ceps[:half]


# --------------------------------------------------------------------------
# MFCC
# --------------------------------------------------------------------------
def _hz2mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel2hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(n_filters: int, nfft: int, fs: float, fmin: float = 0.0, fmax: float | None = None):
    """Trojúhelníková banka melovských filtrů, tvar (n_filters, nfft//2+1)."""
    fmax = fmax or fs / 2
    mel_pts = np.linspace(_hz2mel(fmin), _hz2mel(fmax), n_filters + 2)
    bins = np.floor((nfft + 1) * _mel2hz(mel_pts) / fs).astype(int)
    fb = np.zeros((n_filters, nfft // 2 + 1))
    for i in range(1, n_filters + 1):
        left, centre, right = bins[i - 1], bins[i], bins[i + 1]
        for k in range(left, centre):
            fb[i - 1, k] = (k - left) / max(centre - left, 1)
        for k in range(centre, right):
            fb[i - 1, k] = (right - k) / max(right - centre, 1)
    return fb


def mfcc(x: np.ndarray, fs: float, n_mfcc: int = 13, n_filters: int = 26,
         frame_ms: float = 32.0, hop_ms: float = 16.0, nfft: int = 512) -> np.ndarray:
    """MFCC koeficienty, tvar (n_rámců, n_mfcc)."""
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])  # pre-emfáze
    size, hop = int(fs * frame_ms / 1000), int(fs * hop_ms / 1000)
    frames = frame_signal(x, size, hop) * np.hamming(size)
    power = np.abs(rfft(frames, nfft, axis=1)) ** 2 / nfft
    energies = np.log(power @ mel_filterbank(n_filters, nfft, fs).T + 1e-12)
    return dct(energies, type=2, axis=1, norm="ortho")[:, :n_mfcc]
