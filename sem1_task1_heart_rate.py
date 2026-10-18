"""Seminární práce I, úloha 1: tepová frekvence z EKG (detekce R vrcholů).

Algoritmus: Pan-Tompkins
  1) pásmová propust 5-15 Hz (potlačení driftu, šumu a vln P/T)
  2) derivace (zvýraznění strmých hran QRS)
  3) umocnění (kladný signál, zvýraznění velkých hodnot)
  4) integrace klouzavým oknem (~150 ms)
  5) adaptivní práh (odhad úrovně signálu SPKI a šumu NPKI), refrakterní perioda,
     odmítnutí T vln podle sklonu, zpětné hledání (search-back) vynechaných tepů
  6) zpřesnění polohy vrcholu v filtrovaném signálu

Použití:
  python sem1_task1_heart_rate.py                    # data streamovaná z PhysioNetu
  python sem1_task1_heart_rate.py --data-dir data    # lokální kopie
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d
from scipy.signal import find_peaks

import physio
from dsp import bandpass, interpolate_nans, moving_average

DRIVE_DB, NSR_DB = "drivedb", "nsrdb"
MAX_GAP_S = 3.0   # delší mezera mezi referenčními anotacemi = neanotovaný úsek


@dataclass(frozen=True)
class PanTompkinsParams:
    band_hz: tuple[float, float] = (5.0, 15.0)
    integration_ms: float = 150.0
    refractory_ms: float = 200.0
    twave_ms: float = 360.0
    search_back_factor: float = 1.66
    refine_ms: float = 100.0
    clip_factor: float = 4.0       # ořez filtrovaného signálu na násobek 99. percentilu
    stall_factor: float = 3.0      # po pauze delší než násobek průměrného RR práh klesá


def detect_r_peaks(ecg: np.ndarray, fs: float, p: PanTompkinsParams = PanTompkinsParams()) -> np.ndarray:
    """Vrací indexy (vzorky) detekovaných R vrcholů."""
    ecg = interpolate_nans(ecg)
    filtered = bandpass(ecg, fs, *p.band_hz)
    # Ořez extrémních artefaktů (pohyb, saturace). Bez něj jediný artefakt zvýší adaptivní
    # práh natolik, že se zbytek záznamu přestane detekovat (drivedb: drive02).
    limit = p.clip_factor * np.percentile(np.abs(filtered), 99)
    filtered = np.clip(filtered, -limit, limit)
    deriv = np.gradient(filtered) * fs
    integrated = moving_average(deriv ** 2, int(p.integration_ms / 1000 * fs))
    slope = maximum_filter1d(np.abs(deriv), size=max(int(0.15 * fs), 1))

    cand, _ = find_peaks(integrated, distance=max(int(p.refractory_ms / 1000 * fs), 1))
    if cand.size == 0:
        return cand
    amp = integrated[cand]

    head = integrated[: int(2 * fs)]
    spki, npki = 0.25 * head.max(), 0.5 * head.mean()
    threshold = lambda: npki + 0.25 * (spki - npki)

    twave = p.twave_ms / 1000 * fs
    accepted: list[int] = []   # indexy do pole cand
    rejected: list[int] = []   # kandidáti zamítnutí od posledního tepu
    rr: list[float] = []

    for k, pos in enumerate(cand):
        # search-back: dlouhá pauza -> zkus vzít nejsilnější dříve zamítnutý kandidát
        if len(rr) >= 3 and pos - cand[accepted[-1]] > p.search_back_factor * np.mean(rr[-8:]):
            pool = [j for j in rejected if amp[j] > 0.5 * threshold()]
            if pool:
                j = max(pool, key=lambda i: amp[i])
                rr.append(cand[j] - cand[accepted[-1]])
                accepted.append(j)
                spki = 0.25 * amp[j] + 0.75 * spki
                rejected = [i for i in rejected if cand[i] > cand[j]]

        # zotavení po artefaktu: dlouho bez detekce -> úroveň signálu SPKI postupně klesá
        last = cand[accepted[-1]] if accepted else 0
        mean_rr = np.mean(rr[-8:]) if rr else 1.5 * fs
        if pos - last > p.stall_factor * mean_rr:
            spki *= 0.7

        if amp[k] > threshold():
            if accepted:
                dt = pos - cand[accepted[-1]]
                if dt < twave and slope[pos] < 0.5 * slope[cand[accepted[-1]]]:
                    npki = 0.125 * amp[k] + 0.875 * npki   # T vlna
                    rejected.append(k)
                    continue
                rr.append(dt)
            accepted.append(k)
            spki = 0.125 * amp[k] + 0.875 * spki
            rejected = []
        else:
            npki = 0.125 * amp[k] + 0.875 * npki
            rejected.append(k)

    # zpřesnění: maximum |filtrovaný signál| v okolí vrcholu integrovaného signálu
    half = int(p.refine_ms / 1000 * fs)
    peaks = []
    for idx in cand[accepted]:
        lo, hi = max(idx - half, 0), min(idx + half + 1, filtered.size)
        peaks.append(lo + int(np.argmax(np.abs(filtered[lo:hi]))))
    peaks = np.unique(peaks)
    keep = np.insert(np.diff(peaks) >= p.refractory_ms / 1000 * fs, 0, True)
    return peaks[keep]


def heart_rate(peaks: np.ndarray, fs: float, n_samples: int) -> dict:
    """Souhrnné ukazatele tepové frekvence z pozic R vrcholů."""
    duration_min = n_samples / fs / 60
    rr = np.diff(peaks) / fs
    inst = 60.0 / rr if rr.size else np.array([np.nan])
    return {
        "duration_min": duration_min,
        "n_peaks": int(peaks.size),
        "bpm_mean": peaks.size / duration_min,          # počet vrcholů / délka v minutách
        "bpm_median": float(np.median(inst)),            # robustnější odhad z RR intervalů
        "bpm_std": float(np.std(inst)),
    }


def annotated_mask(detected: np.ndarray, reference: np.ndarray, tol: int, max_gap: int) -> np.ndarray:
    """True pro detekce, které leží v úseku pokrytém referenčními anotacemi.

    Úseky bez anotací (před první, za poslední a mezery delší než max_gap vzorků, typicky
    nekvalitní signál, který experti neanotovali) nelze vyhodnotit, detekce v nich se
    proto nepočítají jako falešně pozitivní.
    """
    keep = (detected >= reference[0] - tol) & (detected <= reference[-1] + tol)
    for a, b in zip(reference[:-1], reference[1:]):
        if b - a > max_gap:
            keep &= ~((detected > a + tol) & (detected < b - tol))
    return keep


def unmatched(detected: np.ndarray, reference: np.ndarray, tol: int) -> np.ndarray:
    """True pro detekce bez anotace v toleranci tol (falešně pozitivní)."""
    idx = np.clip(np.searchsorted(reference, detected), 1, len(reference) - 1)
    dist = np.minimum(np.abs(detected - reference[idx - 1]), np.abs(detected - reference[idx]))
    return dist > tol


def match_beats(detected: np.ndarray, reference: np.ndarray, tol: int) -> tuple[int, int, int]:
    """Párování detekcí s anotacemi v toleranci tol vzorků -> (TP, FP, FN)."""
    i = j = tp = fp = fn = 0
    while i < len(detected) and j < len(reference):
        if abs(detected[i] - reference[j]) <= tol:
            tp += 1; i += 1; j += 1
        elif detected[i] < reference[j]:
            fp += 1; i += 1
        else:
            fn += 1; j += 1
    return tp, fp + len(detected) - i, fn + len(reference) - j


# --------------------------------------------------------------------------
def analyse_drivedb(data_dir, out: Path) -> pd.DataFrame:
    rows = []
    for rec in physio.list_records(DRIVE_DB, data_dir):
        try:
            ecg, fs = physio.read_signal(rec, DRIVE_DB, "ECG", data_dir)
        except KeyError as exc:
            print("Přeskakuji:", exc)
            continue
        peaks = detect_r_peaks(ecg, fs)
        rows.append({"record": rec, "fs": fs, **heart_rate(peaks, fs, len(ecg))})
        print(f"{rec}: {rows[-1]['bpm_mean']:.1f} tepů/min ({peaks.size} R vrcholů)")
        if len(rows) == 1:
            plot_detection(ecg, fs, peaks, out / "drivedb_ukazka_detekce.png", f"{rec}: detekce R vrcholů")

    df = pd.DataFrame(rows)
    df.round(2).to_csv(out / "drivedb_tepova_frekvence.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df["record"], df["bpm_mean"], yerr=df["bpm_std"], capsize=3, color="tab:red", alpha=0.8)
    ax.set(xlabel="Měření", ylabel="Tepová frekvence [tepů/min]", title="Tepová frekvence při jednotlivých jízdách (drivedb)")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "drivedb_tepova_frekvence.png", dpi=150); plt.close(fig)
    return df


def analyse_nsrdb(data_dir, out: Path, minutes: float, tol_ms: float) -> pd.DataFrame:
    rows = []
    for rec in physio.list_records(NSR_DB, data_dir):
        ecg, fs = physio.read_signal(rec, NSR_DB, "ECG1", data_dir, seconds=minutes * 60)
        ref = physio.read_beat_annotations(rec, NSR_DB, data_dir, sampto=len(ecg))
        peaks = detect_r_peaks(ecg, fs)
        tol = int(tol_ms / 1000 * fs)
        keep = annotated_mask(peaks, ref, tol, int(MAX_GAP_S * fs))
        n_excluded = int((~keep).sum())
        peaks = peaks[keep]
        gaps = np.diff(ref); gaps = gaps[gaps > MAX_GAP_S * fs]
        eval_min = ((ref[-1] - ref[0]) - gaps.sum()) / fs / 60      # délka vyhodnocených úseků
        tp, fp, fn = match_beats(peaks, ref, tol)
        hr_det = len(peaks) / eval_min
        hr_ref = len(ref) / eval_min
        rows.append({
            "record": rec, "ref_beats": len(ref), "detected": len(peaks), "excluded": n_excluded, "TP": tp, "FP": fp, "FN": fn,
            "sensitivity_%": 100 * tp / max(tp + fn, 1), "PPV_%": 100 * tp / max(tp + fp, 1),
            "bpm_ref": hr_ref, "bpm_detected": hr_det, "bpm_error": hr_det - hr_ref,
        })
        print(f"{rec}: Se={rows[-1]['sensitivity_%']:.2f} %, PPV={rows[-1]['PPV_%']:.2f} %, chyba BPM={rows[-1]['bpm_error']:+.2f}")

    df = pd.DataFrame(rows)
    tp, fp, fn = df["TP"].sum(), df["FP"].sum(), df["FN"].sum()
    total = {"record": "CELKEM", "ref_beats": df["ref_beats"].sum(), "detected": df["detected"].sum(), "excluded": df["excluded"].sum(),
             "TP": tp, "FP": fp, "FN": fn, "sensitivity_%": 100 * tp / (tp + fn), "PPV_%": 100 * tp / (tp + fp),
             "bpm_ref": df["bpm_ref"].mean(), "bpm_detected": df["bpm_detected"].mean(),
             "bpm_error": df["bpm_error"].abs().mean()}
    df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
    df.round(2).to_csv(out / "nsrdb_uspesnost.csv", index=False)

    body = df.iloc[:-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(body["bpm_ref"], body["bpm_detected"], color="tab:blue")
    lim = [body[["bpm_ref", "bpm_detected"]].min().min() - 3, body[["bpm_ref", "bpm_detected"]].max().max() + 3]
    axes[0].plot(lim, lim, "k--", lw=1)
    axes[0].set(xlabel="Referenční BPM (anotace)", ylabel="Detekované BPM", title="Shoda tepové frekvence (nsrdb)")
    axes[1].bar(body["record"], body["sensitivity_%"], width=0.4, label="Senzitivita", align="edge")
    axes[1].bar(body["record"], body["PPV_%"], width=-0.4, label="PPV", align="edge")
    axes[1].set(ylim=(max(0, min(body[["sensitivity_%", "PPV_%"]].min()) - 2), 100.5), ylabel="[%]", title="Úspěšnost detekce R vrcholů")
    axes[1].tick_params(axis="x", rotation=60); axes[1].legend()
    fig.tight_layout(); fig.savefig(out / "nsrdb_uspesnost.png", dpi=150); plt.close(fig)
    return df


def plot_detection(ecg, fs, peaks, path: Path, title: str, seconds: float = 10.0):
    n = int(seconds * fs)
    t = np.arange(n) / fs
    p = peaks[peaks < n]
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(t, interpolate_nans(ecg)[:n], lw=0.8)
    ax.plot(p / fs, interpolate_nans(ecg)[p], "rv", label="R vrcholy")
    ax.set(xlabel="t [s]", ylabel="U [mV]", title=title); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_record(rec: str, data_dir, out: Path, minutes: float = 30.0, tol_ms: float = 100.0, window_s: float = 20.0):
    """Ladicí graf jednoho záznamu.

    drivedb: začátek, střed a konec záznamu.
    nsrdb:   okna kolem falešně pozitivních detekcí (po vyřazení neanotovaných úseků),
             případně začátek, střed a konec, pokud žádné nejsou.
    """
    for db, channel in ((DRIVE_DB, "ECG"), (NSR_DB, "ECG1")):
        if rec not in physio.list_records(db, data_dir):
            continue
        hdr = physio.read_header(rec, db, data_dir)
        seconds = min(minutes * 60, hdr.sig_len / hdr.fs) if db == NSR_DB else None
        ecg, fs = physio.read_signal(rec, db, channel, data_dir, seconds=seconds)
        peaks = detect_r_peaks(ecg, fs)
        x = interpolate_nans(ecg)
        n = int(window_s * fs)
        ref, starts, note = None, None, ""
        if db == NSR_DB:
            ref = physio.read_beat_annotations(rec, db, data_dir, sampto=len(ecg))
            tol = int(tol_ms / 1000 * fs)
            keep = annotated_mask(peaks, ref, tol, int(MAX_GAP_S * fs))
            fp = peaks[keep][unmatched(peaks[keep], ref, tol)]
            note = f", FP: {len(fp)}, vyřazeno (neanotováno): {int((~keep).sum())}"
            if len(fp):
                starts = [max(int(fp[int(q * (len(fp) - 1))]) - n // 2, 0) for q in (0.1, 0.5, 0.9)]
        if starts is None:
            starts = [0, max(len(x) // 2 - n // 2, 0), max(len(x) - n, 0)]
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        for ax, start in zip(axes, starts):
            start = min(start, max(len(x) - n, 0))
            t = np.arange(start, min(start + n, len(x))) / fs
            ax.plot(t, x[start:start + n], lw=0.7)
            p = peaks[(peaks >= start) & (peaks < start + n)]
            ax.plot(p / fs, x[p], "rv", label="detekce")
            if ref is not None:
                r = ref[(ref >= start) & (ref < start + n)]
                ax.plot(r / fs, x[r], "go", mfc="none", ms=9, label="anotace")
            ax.set_xlabel("t [s]")
        axes[0].legend()
        axes[0].set_title(f"{rec} ({db}): {len(peaks)} detekcí, fs = {fs:g} Hz, rozsah {np.nanmin(ecg):.2f} až {np.nanmax(ecg):.2f}{note}")
        fig.tight_layout(); path = out / f"ladeni_{rec}.png"; fig.savefig(path, dpi=130); plt.close(fig)
        print("Uloženo:", path)
        return
    raise SystemExit(f"Záznam {rec} nebyl nalezen.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None, help="Lokální složka s databázemi (jinak streamování z PhysioNetu)")
    parser.add_argument("--out", default="outputs/sem1_task1")
    parser.add_argument("--nsrdb-minutes", type=float, default=30.0, help="Délka testovaného úseku každého záznamu nsrdb")
    parser.add_argument("--tolerance-ms", type=float, default=100.0, help="Tolerance párování detekce a anotace")
    parser.add_argument("--plot", metavar="ZÁZNAM", help="Jen vykreslit detekci pro jeden záznam (např. drive02, 16272) a skončit")
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.plot:
        plot_record(args.plot, args.data_dir, out, args.nsrdb_minutes, args.tolerance_ms)
        return

    print("=== drivedb ===")
    analyse_drivedb(args.data_dir, out)
    print("\n=== nsrdb (validace) ===")
    result = analyse_nsrdb(args.data_dir, out, args.nsrdb_minutes, args.tolerance_ms)
    print("\n", result.round(2).to_string(index=False))
    print(f"\nVýstupy uloženy do {out}/")


if __name__ == "__main__":
    main()