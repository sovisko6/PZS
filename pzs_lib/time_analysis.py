import numpy as np
import scipy.signal as signal

def calculate_energy(sig):
    """Vypočítá energii diskrétního signálu: suma(|x|^2)."""
    return np.sum(np.abs(sig)**2)

def calculate_power(sig):
    """Vypočítá střední výkon signálu: 1/N * suma(|x|^2)."""
    return np.mean(np.abs(sig)**2)

def statistics(sig):
    """Vrátí základní statistiky: mean, var, std."""
    return {
        "mean": np.mean(sig),
        "variance": np.var(sig),
        "std_dev": np.std(sig),
        "min": np.min(sig),
        "max": np.max(sig)
    }

def custom_convolution(sig1, sig2, mode='full'):
    """Wrapper pro konvoluci."""
    return np.convolve(sig1, sig2, mode=mode)

def covariance(sig1, sig2):
    """
    Vypočítá kovarianci dvou signálů.
    Kovariance měří, jak moc se dva signály mění společně.
    
    cov(X,Y) = E[(X - E[X])(Y - E[Y])]
    """
    mean1 = np.mean(sig1)
    mean2 = np.mean(sig2)
    return np.mean((sig1 - mean1) * (sig2 - mean2))

def autocorrelation(sig):
    """Vypočítá autokorelaci signálu."""
    # Použijeme korelaci signálu se sebou samým
    result = np.correlate(sig, sig, mode='full')
    return result[result.size // 2:] # Vracíme jen kladnou část zpoždění (lags)

def cross_correlation(sig1, sig2):
    """Vypočítá křížovou korelaci dvou signálů."""
    return np.correlate(sig1, sig2, mode='full')

def find_peaks(sig, height=None, distance=None):
    """
    Najde indexy vrcholů v signálu (např. R-vlny v EKG).
    Wrapper pro scipy.signal.find_peaks.
    
    Args:
        sig: Vstupní signál
        height: Minimální výška vrcholu (práh)
        distance: Minimální vzdálenost mezi vrcholy ve vzorcích
    Returns:
        peaks: Pole indexů, kde jsou vrcholy
    """
    peaks, _ = signal.find_peaks(sig, height=height, distance=distance)
    return peaks

def resample_signal(sig, original_fs, target_fs):
    """
    Převzorkuje signál na novou frekvenci.
    Nutné pro korelaci signálů s různým fs (jak je v zadání I).
    """
    if original_fs == target_fs:
        return sig
    
    duration = len(sig) / original_fs
    target_samples = int(duration * target_fs)
    
    return signal.resample(sig, target_samples)

def normalize_signal(sig):
    """
    Normalizuje signál do rozsahu -1 až 1 (nebo 0 až 1).
    Hodí se pro porovnávání tvarů (korelace).
    """
    # Odstranění stejnosměrné složky (centrování)
    sig_centered = sig - np.mean(sig)
    # Normalizace podle maxima absolutní hodnoty
    return sig_centered / np.max(np.abs(sig_centered))


def ensemble_averaging(signals):
    """
    Kumulační technika - průměrování ansámblů (ensemble averaging).
    Posílení signálu potlačením nekoreovaného šumu.
    
    Týden 6: "Kumulační techniky pro posílení signálu v šumu"
    
    Args:
        signals: List nebo 2D numpy array, kde každý řádek je jedno měření
    Returns:
        averaged_signal: Průměrovaný signál
    """
    signals = np.array(signals)
    return np.mean(signals, axis=0)


def moving_average_cumulative(sig, window_size):
    """
    Kumulační klouzavý průměr pro potlačení šumu.
    
    Týden 6: "Kumulační techniky pro posílení signálu v šumu"
    """
    cumsum = np.cumsum(np.insert(sig, 0, 0)) 
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


def correlation_peak_detection(sig, template):
    """
    Korelační technika pro detekci vzoru v signálu.
    
    Týden 7: "Korelační techniky pro posílení signálu v šumu"
    
    Args:
        sig: Hlavní signál (např. EKG)
        template: Vzor, který hledáme (např. R-vlna)
    Returns:
        correlation: Korelační funkce
        peaks: Indexy detekovaných výskytů vzoru
    """
    # Normalizace pro lepší detekci
    sig_norm = normalize_signal(sig)
    template_norm = normalize_signal(template)
    
    # Křížová korelace
    correlation = np.correlate(sig_norm, template_norm, mode='same')
    
    # Detekce píků v korelaci
    threshold = 0.7 * np.max(correlation)
    peaks = find_peaks(correlation, height=threshold, distance=len(template)//2)
    
    return correlation, peaks


def matched_filter(sig, template):
    """
    Matched filter - optimální filtr pro detekci známého signálu v šumu.
    
    Týden 7: "Korelační techniky pro posílení signálu v šumu"
    
    Matematicky ekvivalentní křížové korelaci s časově obráceným vzorem.
    """
    # Časově obrácený template
    template_reversed = template[::-1]
    # Konvoluce = korelace s obráceným signálem
    return np.convolve(sig, template_reversed, mode='same')


def calculate_hnr(sig, fs, f0_min=75, f0_max=500, frame_length=0.03, num_frames=5):
    """
    Vypočítá HNR (Harmonic-to-Noise Ratio) pomocí autokorelace.
    
    Týden 5-6: "Časová analýza signálu"
    
    HNR měří poměr harmonických (periodických) složek k šumu.
    Vysoké HNR (>10 dB) = čistý harmonický signál (zdravý hlas)
    Nízké HNR (<5 dB) = šumový signál (patologický hlas)
    
    Args:
        sig: Vstupní signál (hlas)
        fs: Vzorkovací frekvence [Hz]
        f0_min: Minimální očekávaná základní frekvence [Hz] (default 75 Hz)
        f0_max: Maximální očekávaná základní frekvence [Hz] (default 500 Hz)
        frame_length: Délka analyzovaného segmentu v sekundách (default 0.03s = 30ms)
        num_frames: Počet segmentů k průměrování (default 5)
    
    Returns:
        hnr: Průměrné Harmonic-to-Noise Ratio v dB
    """
    # VYLEPŠENÍ: Místo 1 segmentu spočítáme HNR pro VÍCE SEGMENTŮ a zprůměrujeme
    # To dá robustnější odhad (menší vliv lokálních artefaktů)
    
    n_samples = int(frame_length * fs)
    sig_length = len(sig)
    
    # Vytvoříme rovnoměrně rozložené segmenty přes celý signál
    # (první čtvrtina, střed-levá, střed, střed-pravá, poslední čtvrtina)
    hnr_values = []
    
    for i in range(num_frames):
        # Pozice středu každého segmentu
        position = (i + 1) / (num_frames + 1)  # 0.17, 0.33, 0.50, 0.67, 0.83
        center = int(sig_length * position)
        
        start_idx = max(0, center - n_samples // 2)
        end_idx = min(sig_length, center + n_samples // 2)
        
        if end_idx - start_idx < n_samples // 2:
            continue  # Přeskočíme příliš krátké segmenty
        
        sig_segment = sig[start_idx:end_idx]
        
        # Vypočítáme autokorelaci pro tento segment
        acf = autocorrelation(sig_segment)
        # Vypočítáme autokorelaci pro tento segment
        acf = autocorrelation(sig_segment)
        
        # Rozsah zpoždění odpovídající f0_min až f0_max
        min_lag = int(fs / f0_max)
        max_lag = int(fs / f0_min)
        
        if max_lag >= len(acf):
            max_lag = len(acf) - 1
        
        if min_lag >= max_lag:
            continue  # Neplatný rozsah
        
        # Najdeme maximum autokorelace v daném rozsahu
        search_range = acf[min_lag:max_lag]
        
        if len(search_range) == 0:
            continue
        
        # ACF[T0] = maximum autokorelace = harmonická složka
        max_acf = np.max(search_range)
        
        # ACF[0] = autokorelace při nulové prodlevě = celková energie signálu
        total_energy = acf[0]
        
        # HNR vzorec: 10 * log10( harmonic / noise )
        noise_energy = total_energy - max_acf
        
        if noise_energy > 0 and max_acf > 0:
            hnr_segment = 10 * np.log10(max_acf / noise_energy)
            hnr_values.append(hnr_segment)
    
    # Vrátíme průměr přes všechny segmenty (robustnější odhad!)
    if len(hnr_values) > 0:
        return np.mean(hnr_values)
    else:
        return 0.0


def calculate_jitter(sig, fs, f0_min=75, f0_max=500):
    """
    Vypočítá Jitter - variabilitu základní frekvence (F0) mezi po sobě jdoucími periodami.
    ORIGINÁLNÍ VERZE s autokorelací a peak detection.
    
    Týden 5-6: "Časová analýza signálu - perturbační analýza"
    
    Jitter měří nestabilitu kmitání hlasivek.
    Vysoký jitter = patologický hlas (nepravidelné kmitání)
    Nízký jitter = zdravý hlas (pravidelné kmitání)
    
    Args:
        sig: Vstupní signál (hlas)
        fs: Vzorkovací frekvence [Hz]
        f0_min: Minimální očekávaná F0 [Hz]
        f0_max: Maximální očekávaná F0 [Hz]
    
    Returns:
        jitter: Jitter v procentech [%]
    """
    from scipy.signal import find_peaks
    
    # Parametry pro hledání period
    min_period = int(fs / f0_max)  # Minimální perioda (samples)
    max_period = int(fs / f0_min)  # Maximální perioda (samples)
    
    # Najdeme píky v signálu (aproximace začátků period)
    # Použijeme robustnější threshold - 30% maxima
    threshold = 0.3 * np.max(np.abs(sig))
    peaks, _ = find_peaks(sig, height=threshold, distance=min_period)
    
    if len(peaks) < 3:
        # Fallback: použijeme autokorelaci pro odhad periody
        autocorr = np.correlate(sig, sig, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Hledáme první maximum autokorelace v platném rozsahu
        search_region = autocorr[min_period:max_period]
        if len(search_region) > 0:
            period_estimate = np.argmax(search_region) + min_period
            # Aproximace jitter z autokorelace
            return 1.0  # Fallback hodnota
        return 0.0
    
    # Spočítáme vzdálenosti mezi po sobě jdoucími píky (periody)
    periods = np.diff(peaks)
    
    # Filtrujeme periody mimo platný rozsah
    valid_periods = periods[(periods >= min_period) & (periods <= max_period)]
    
    if len(valid_periods) < 2:
        return 0.0
    
    # Jitter = průměrná absolutní diference po sobě jdoucích period / průměrná perioda
    # (v procentech)
    period_diffs = np.abs(np.diff(valid_periods))
    mean_period = np.mean(valid_periods)
    
    if mean_period > 0:
        jitter = (np.mean(period_diffs) / mean_period) * 100
    else:
        jitter = 0.0
    
    return jitter


def calculate_shimmer(sig, fs, f0_min=75, f0_max=500):
    """
    Vypočítá Shimmer - variabilitu amplitudy mezi po sobě jdoucími periodami.
    ORIGINÁLNÍ VERZE s autokorelací a period-matched RMS.
    
    Týden 5-6: "Časová analýza signálu - perturbační analýza"
    
    Shimmer měří nestabilitu amplitudy hlasu.
    Vysoký shimmer = patologický hlas (kolísání hlasitosti)
    Nízký shimmer = zdravý hlas (stabilní hlasitost)
    
    Args:
        sig: Vstupní signál (hlas)
        fs: Vzorkovací frekvence [Hz]
        f0_min: Minimální očekávaná F0 [Hz]
        f0_max: Maximální očekávaná F0 [Hz]
    
    Returns:
        shimmer: Shimmer v procentech [%]
    """
    from scipy.signal import find_peaks
    
    # Parametry pro hledání period
    min_period = int(fs / f0_max)
    max_period = int(fs / f0_min)
    
    # Najdeme píky v signálu
    threshold = 0.3 * np.max(np.abs(sig))
    peaks, _ = find_peaks(sig, height=threshold, distance=min_period)
    
    if len(peaks) < 3:
        # Fallback: použijeme autokorelaci pro odhad periody
        autocorr = np.correlate(sig, sig, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        search_region = autocorr[min_period:max_period]
        if len(search_region) > 0:
            period_estimate = np.argmax(search_region) + min_period
            # Rozdělíme na segmenty
            num_segments = len(sig) // period_estimate
            if num_segments < 2:
                return 0.0
            
            amplitudes = []
            for i in range(num_segments):
                start = i * period_estimate
                end = start + period_estimate
                if end > len(sig):
                    break
                segment = sig[start:end]
                amp = np.sqrt(np.mean(segment ** 2))  # RMS
                amplitudes.append(amp)
            
            amplitudes = np.array(amplitudes)
            if len(amplitudes) > 1 and np.mean(amplitudes) > 0:
                amp_diffs = np.abs(np.diff(amplitudes))
                shimmer = (np.mean(amp_diffs) / np.mean(amplitudes)) * 100
                return shimmer
        return 0.0
    
    # Spočítáme RMS amplitudu pro každou periodu (mezi po sobě jdoucími píky)
    amplitudes = []
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]
        
        # Ověříme délku periody
        period_length = end - start
        if period_length < min_period or period_length > max_period:
            continue
        
        segment = sig[start:end]
        # RMS amplitude pro tuto periodu
        amp = np.sqrt(np.mean(segment ** 2))
        amplitudes.append(amp)
    
    if len(amplitudes) < 2:
        return 0.0
    
    amplitudes = np.array(amplitudes)
    
    # Shimmer = průměrná absolutní diference po sobě jdoucích amplitud / průměrná amplituda
    amp_diffs = np.abs(np.diff(amplitudes))
    mean_amp = np.mean(amplitudes)
    
    if mean_amp > 0:
        shimmer = (np.mean(amp_diffs) / mean_amp) * 100
    else:
        shimmer = 0.0
    
    return shimmer


def calculate_zcr(sig):
    """
    Vypočítá Zero-Crossing Rate (ZCR) - míru průchodů signálu nulou.
    
    Týden 5-6: "Časová analýza signálu"
    
    ZCR měří frekvenci změn znaménka signálu.
    Vysoký ZCR = šumový nebo vysokofrekvenční signál
    Nízký ZCR = nízko-frekvenční nebo harmonický signál
    
    Args:
        sig: Vstupní signál
    
    Returns:
        zcr: Zero-crossing rate (počet průchodů / počet vzorků)
    """
    # Počet změn znaménka
    zero_crossings = np.sum(np.abs(np.diff(np.sign(sig)))) / 2
    
    # Normalizace délkou signálu
    zcr = zero_crossings / len(sig)
    
    return zcr


def calculate_energy_variability(sig, fs, frame_length=0.03, hop_length=0.015):
    """
    Vypočítá variabilitu krátkodobé energie signálu.
    
    Týden 5-6: "Časová analýza signálu"
    
    Měří stabilitu hlasitosti v čase.
    Vysoká variabilita = nestabilní hlas (patologický)
    Nízká variabilita = stabilní hlas (zdravý)
    
    Args:
        sig: Vstupní signál
        fs: Vzorkovací frekvence
        frame_length: Délka okna v sekundách (default 30ms)
        hop_length: Posun okna v sekundách (default 15ms)
    
    Returns:
        energy_var: Koeficient variace energie (std/mean)
    """
    frame_samples = int(frame_length * fs)
    hop_samples = int(hop_length * fs)
    
    energies = []
    
    # Sliding window přes signál
    for start in range(0, len(sig) - frame_samples, hop_samples):
        frame = sig[start:start + frame_samples]
        energy = np.sum(frame ** 2)
        energies.append(energy)
    
    if len(energies) == 0:
        return 0.0
    
    energies = np.array(energies)
    
    # Koeficient variace = std / mean
    mean_energy = np.mean(energies)
    std_energy = np.std(energies)
    
    if mean_energy > 0:
        energy_var = std_energy / mean_energy
    else:
        energy_var = 0.0
    
    return energy_var