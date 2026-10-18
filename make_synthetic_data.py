"""Vygeneruje syntetická data ve formátu WFDB pro offline ověření funkčnosti kódu.

    python make_synthetic_data.py --out synthetic
    python sem1_task1_heart_rate.py --data-dir synthetic --out outputs_test/t1 --nsrdb-minutes 2
    python sem1_task2_correlation.py --data-dir synthetic --out outputs_test/t2 --minutes 5
    python sem2_voiced_classification.py --data-dir synthetic --out outputs_test/s2

Výsledky na syntetických datech nemají žádnou vypovídací hodnotu pro seminární práci.
"""
import argparse
from pathlib import Path

import numpy as np
import wfdb

RNG = np.random.default_rng(0)


def ecg(fs, seconds, bpm):
    t = np.arange(int(fs * seconds)) / fs
    beats = np.cumsum(np.clip(RNG.normal(60 / bpm, 0.03, int(seconds * bpm / 60) + 5), 0.4, None))
    beats = beats[beats < seconds - 0.5]
    x = sum(np.exp(-((t - b) ** 2) / (2 * 0.012 ** 2)) - 0.1 * np.exp(-((t - b + 0.04) ** 2) / (2 * 0.01 ** 2))
            + 0.25 * np.exp(-((t - b - 0.25) ** 2) / (2 * 0.04 ** 2)) for b in beats)
    return x + 0.03 * RNG.normal(size=t.size) + 0.2 * np.sin(2 * np.pi * 0.2 * t), (beats * fs).astype(int)


def write_drivedb(root, n=3):
    d = root / "drivedb"; d.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        x, _ = ecg(496, 120, 60 + 8 * i)
        other = RNG.normal(size=int(15.5 * 120))
        wfdb.wrsamp(f"drive{i:02d}", fs=15.5, units=["mV", "mV"], sig_name=["ECG", "foot GSR"],
                    fmt=["16", "16"], adc_gain=[200, 200], baseline=[0, 0], samps_per_frame=[32, 1],
                    e_p_signal=[x, other], write_dir=str(d))
    (d / "RECORDS").write_text("\n".join(f"drive{i:02d}" for i in range(1, n + 1)))


def write_nsrdb(root, n=3):
    d = root / "nsrdb"; d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        name = f"1{6265 + i}"
        x, beats = ecg(128, 300, 65 + 5 * i)
        y, _ = ecg(128, 300, 65)
        wfdb.wrsamp(name, fs=128, units=["mV", "mV"], sig_name=["ECG1", "ECG2"], p_signal=np.c_[x, y],
                    fmt=["16", "16"], write_dir=str(d))
        wfdb.wrann(name, "atr", beats, ["N"] * len(beats), write_dir=str(d))


def write_charisdb(root, n=4, fs=50):
    d = root / "charisdb"; d.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        sig, _ = ecg(fs, 600, 70 + 5 * i)
        t = np.arange(sig.size) / fs
        abp = 80 + 20 * np.sin(2 * np.pi * (1 + 0.05 * i) * t) + RNG.normal(size=t.size)
        icp = 10 + 3 * np.sin(2 * np.pi * (1 + 0.05 * i) * t - 0.5) + 0.2 * RNG.normal(size=t.size)
        wfdb.wrsamp(f"chari{i:03d}", fs=fs, units=["mV", "mmHg", "mmHg"], sig_name=["ECG", "ABP", "ICP"],
                    p_signal=np.c_[sig, abp, icp], fmt=["16"] * 3, write_dir=str(d))


def voice(fs, seconds, f0, jitter, shimmer, noise):
    t, phase, out, amp = 0.0, 0.0, [], 1.0
    n = int(fs * seconds)
    x = np.zeros(n)
    pos = 0.0
    while pos < n:
        period = fs / f0 * (1 + jitter * RNG.normal())
        a = amp * (1 + shimmer * RNG.normal())
        k = np.arange(int(period))
        x[int(pos):int(pos) + len(k)] += (a * sum(np.sin(2 * np.pi * h * k / period) / h for h in range(1, 12)))[: n - int(pos)]
        pos += period
    x += noise * RNG.normal(size=n)
    x *= np.minimum(1, np.minimum(np.arange(n), n - np.arange(n)) / (0.1 * fs))
    return x / np.max(np.abs(x)) * 0.8


def write_voiced(root):
    d = root / "voiced"; d.mkdir(parents=True, exist_ok=True)
    groups = {"healthy": (30, 0.005, 0.03, 0.02), "hyperkinetic dysphonia": (25, 0.02, 0.10, 0.15),
              "hypokinetic dysphonia": (25, 0.012, 0.06, 0.30), "reflux laryngitis": (12, 0.008, 0.08, 0.08),
              "vocal fold nodules": (3, 0.03, 0.12, 0.2)}
    k, names = 1, []
    for diag, (count, jit, shim, noise) in groups.items():
        for _ in range(count):
            x = voice(8000, 5, RNG.uniform(110, 220), jit, shim, noise)
            name = f"voice{k:03d}"; names.append(name); k += 1
            wfdb.wrsamp(name, fs=8000, units=["V"], sig_name=["voice"], p_signal=x[:, None], fmt=["16"],
                        comments=["<age>: 30", "<sex>: f", f"<diagnosis>: {diag}"], write_dir=str(d))
    (d / "RECORDS").write_text("\n".join(names))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="synthetic")
    root = Path(parser.parse_args().out)
    for fn in (write_drivedb, write_nsrdb, write_charisdb, write_voiced):
        fn(root)
    print("Hotovo:", root)
