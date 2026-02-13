import numpy as np

def dft_slow(x):
    """
    Vypočítá Diskrétní Fourierovu Transformaci (DFT) podle definice.
    Vhodné pro pochopení principu (jako v Tyden8.ipynb), ale pomalé pro dlouhé signály.
    """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    # Matice exponenciál: e^(-i * 2*pi * k * n / N)
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def get_frequency_spectrum(sig, fs):
    """
    Vrátí frekvenční osu a magnitudu spektra pomocí rychlé FFT.
    
    Returns:
        freqs: Osa frekvencí (Hz)
        magnitude: Amplituda spektra (normalizovaná)
    """
    N = len(sig)
    # FFT výpočet
    fft_vals = np.fft.fft(sig)
    
    # Frekvenční osa
    freqs = np.fft.fftfreq(N, 1/fs)
    
    # Vezmeme jen první polovinu (pozitivní frekvence)
    half_N = N // 2
    magnitude = np.abs(fft_vals[:half_N]) * (2.0 / N) # Normalizace amplitudy
    freqs = freqs[:half_N]
    
    return freqs, magnitude


def compute_real_cepstrum(sig):
    """
    Vypočítá reálné kepstrum signálu.
    Matematika: IFFT( log( |FFT(x)| ) )
    """
    # 1. Hammingovo okno (důležité pro vyhlazení okrajů u hlasu)
    window = np.hamming(len(sig))
    sig_windowed = sig * window
    
    # 2. FFT
    spectrum = np.fft.fft(sig_windowed)
    
    # 3. Logaritmus magnitudy
    log_spectrum = np.log(np.abs(spectrum) + 1e-10) # epsilon proti log(0)
    
    # 4. IFFT -> Kepstrum
    cepstrum_vals = np.fft.ifft(log_spectrum).real
    
    return cepstrum_vals

def analyze_voice_features(cepstrum_vals, fs, f_min=50, f_max=400):
    """
    Vytáhne z kepstra základní frekvenci (F0) a výraznost píku (CPP).
    
    Args:
        cepstrum_vals: Vypočítané kepstrum
        fs: Vzorkovací frekvence
        f_min, f_max: Rozsah frekvencí, kde hledáme lidský hlas (Hz)
        
    Returns:
        f0: Základní frekvence (Hz)
        cpp: Výška píku (Cepstral Peak Prominence) - určuje kvalitu hlasu
    """
    # Přepočet frekvence na quefrency (indexy)
    # Quefrency = fs / frekvence
    # Pozor: Nízká frekvence = Vysoká quefrency index
    min_index = int(fs / f_max) 
    max_index = int(fs / f_min)
    
    # Ochrana proti indexům mimo pole
    max_index = min(max_index, len(cepstrum_vals) // 2)
    
    # Vyřízneme jen tu část kepstra, kde může být hlas
    valid_region = cepstrum_vals[min_index:max_index]
    
    if len(valid_region) == 0:
        return 0.0, 0.0

    # Najdeme maximum v tomto regionu
    peak_idx_local = np.argmax(valid_region)
    cpp_value = valid_region[peak_idx_local]
    
    # Přepočteme lokální index zpět na globální index a pak na frekvenci
    peak_idx_global = min_index + peak_idx_local
    f0 = fs / peak_idx_global
    
    return f0, cpp_value


def analyze_quefrency_width(cepstrum_vals, fs, f0, width_threshold=0.5):
    """
    Analyzuje šířku kepstrálního píku - měří stabilitu periody.
    
    Týden 12: "Kepstrální analýza - quefrency domain"
    
    Úzký pík = stabilní perioda (zdravý hlas)
    Široký pík = variabilní perioda (patologický hlas, jitter)
    
    Args:
        cepstrum_vals: Vypočítané kepstrum
        fs: Vzorkovací frekvence
        f0: Základní frekvence (Hz) - zjištěná z analyze_voice_features
        width_threshold: Práh pro měření šířky (default 0.5 = šířka v polovině výšky)
    
    Returns:
        peak_width: Šířka píku v quefrency jednotkách (vzorky)
        width_ratio: Poměr šířka/výška (normalizovaný)
    """
    if f0 == 0:
        return 0.0, 0.0
    
    # Index píku odpovídající F0
    peak_idx = int(fs / f0)
    
    if peak_idx >= len(cepstrum_vals) or peak_idx < 1:
        return 0.0, 0.0
    
    peak_height = cepstrum_vals[peak_idx]
    
    if peak_height <= 0:
        return 0.0, 0.0
    
    # Hledáme šířku v polovině výšky (FWHM - Full Width at Half Maximum)
    half_height = peak_height * width_threshold
    
    # Hledáme vlevo od píku
    left_idx = peak_idx
    while left_idx > 0 and cepstrum_vals[left_idx] > half_height:
        left_idx -= 1
    
    # Hledáme vpravo od píku
    right_idx = peak_idx
    while right_idx < len(cepstrum_vals) - 1 and cepstrum_vals[right_idx] > half_height:
        right_idx += 1
    
    # Šířka píku
    peak_width = right_idx - left_idx
    
    # Normalizovaný poměr (větší = širší = horší)
    width_ratio = peak_width / peak_height if peak_height > 0 else 0.0
    
    return peak_width, width_ratio


def spectral_centroid(sig, fs):
    """
    Vypočítá spektrální centroid (těžiště frekvencí).
    
    Týden 11: "Frekvenční analýza periodických i obecných signálů"
    
    Centroid = suma(f * magnitude) / suma(magnitude)
    
    Fyzikální význam: "Jas" zvuku.
    - Vyšší hodnota = světlejší, více vysokých frekvencí
    - Nižší hodnota = temnější zvuk
    
    Zdravé hlasy mají stabilnější spektrální centroid.
    """
    freqs, magnitude = get_frequency_spectrum(sig, fs)
    
    # Spektrální centroid
    centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
    
    return centroid


def spectral_flatness(sig, fs):
    """
    Vypočítá spektrální ploškost (flatness).
    
    Týden 11: "Frekvenční analýza periodických i obecných signálů"
    
    Flatness = geometrický průměr / aritmetický průměr
    
    - Hodnota blízko 1 = bílý šum (ploché spektrum)
    - Hodnota blízko 0 = harmonický signál (hřebínek)
    
    Zdravé hlasy mají nízkou flatness (harmonické).
    Patologické hlasy mají vyšší flatness (více šumu).
    """
    freqs, magnitude = get_frequency_spectrum(sig, fs)
    
    # Geometrický průměr
    magnitude_safe = magnitude + 1e-10  # Ochrana před log(0)
    geom_mean = np.exp(np.mean(np.log(magnitude_safe)))
    
    # Aritmetický průměr
    arith_mean = np.mean(magnitude)
    
    # Flatness
    flatness = geom_mean / (arith_mean + 1e-10)
    
    return flatness


def spectral_rolloff(sig, fs, rolloff_percent=0.85):
    """
    Vypočítá spektrální rolloff - frekvenci, pod kterou je X% energie spektra.
    
    Týden 11: Frekvenční charakteristika signálu.
    
    Args:
        rolloff_percent: Procento energie (typicky 0.85 = 85%)
    """
    freqs, magnitude = get_frequency_spectrum(sig, fs)
    
    # Energie spektra
    power_spectrum = magnitude ** 2
    cumulative_power = np.cumsum(power_spectrum)
    total_power = cumulative_power[-1]
    
    # Najdi frekvenci, kde je X% energie
    rolloff_threshold = rolloff_percent * total_power
    rolloff_idx = np.where(cumulative_power >= rolloff_threshold)[0]
    
    if len(rolloff_idx) > 0:
        return freqs[rolloff_idx[0]]
    else:
        return freqs[-1]


def spectral_bandwidth(sig, fs):
    """
    Vypočítá spektrální šířku pásma (bandwidth).
    
    Týden 11: Frekvenční charakteristika signálu.
    
    Měří "rozptyl" frekvencí kolem centroidu.
    """
    freqs, magnitude = get_frequency_spectrum(sig, fs)
    
    centroid = spectral_centroid(sig, fs)
    
    # Bandwidth jako vážená směrodatná odchylka
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitude) / (np.sum(magnitude) + 1e-10))
    
    return bandwidth


def hilbert_transform(sig):
    """
    Vypočítá Hilbertovu transformaci signálu.
    
    Týden 12: "Hilbertova transformace, zpracování fyziologického signálu"
    
    Používá se pro:
    - Výpočet okamžité amplitudy (envelope)
    - Výpočet okamžité fáze
    - Výpočet okamžité frekvence
    
    Returns:
        analytic_signal: Komplexní analytický signál
        envelope: Obálka signálu (okamžitá amplituda)
        instantaneous_phase: Okamžitá fáze
    """
    # Scipy má implementovanou Hilbertovu transformaci
    from scipy.signal import hilbert
    
    analytic_signal = hilbert(sig)
    envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    
    return analytic_signal, envelope, instantaneous_phase


def instantaneous_frequency(sig, fs):
    """
    Vypočítá okamžitou frekvenci pomocí Hilbertovy transformace.
    
    Týden 12: "Komplexní zpracování signálu v časově-frekvenční oblasti"
    
    Args:
        sig: Vstupní signál
        fs: Vzorkovací frekvence
    Returns:
        inst_freq: Okamžitá frekvence v Hz
    """
    _, _, phase = hilbert_transform(sig)
    
    # Okamžitá frekvence = derivace fáze / (2*pi)
    inst_freq = np.diff(phase) / (2.0 * np.pi) * fs
    
    # Přidáme první hodnotu pro zachování délky
    inst_freq = np.insert(inst_freq, 0, inst_freq[0])
    
    return inst_freq


def spectral_entropy(sig, fs):
    """
    Vypočítá spektrální entropii - míru "chaosu" ve frekvenčním spektru.
    
    Týden 10-11: "Fourierova analýza - spektrální charakteristiky"
    
    Entropie měří, jak rovnoměrně je rozložena energie ve spektru.
    
    Nízká entropie = energie koncentrovaná v několika frekvencích (harmonický signál)
    Vysoká entropie = energie rozložená rovnoměrně (šumový signál)
    
    Zdravý hlas: nízká entropie (čisté harmonické)
    Patologický hlas: vyšší entropie (více šumu)
    
    Args:
        sig: Vstupní signál
        fs: Vzorkovací frekvence
    
    Returns:
        entropy: Spektrální entropie (0 = úplný řád, max = úplný chaos)
    """
    freqs, magnitude = get_frequency_spectrum(sig, fs)
    
    # Normalizace spektra na pravděpodobnostní distribuci
    power_spectrum = magnitude ** 2
    total_power = np.sum(power_spectrum)
    
    if total_power == 0:
        return 0.0
    
    # Pravděpodobnostní distribuce
    prob_dist = power_spectrum / total_power
    
    # Shannon entropie: H = -sum(p * log2(p))
    # Ignorujeme nulové hodnoty (0 * log(0) = 0 podle konvence)
    prob_dist_nonzero = prob_dist[prob_dist > 0]
    
    entropy = -np.sum(prob_dist_nonzero * np.log2(prob_dist_nonzero))
    
    # Normalizace maximální entropií (log2(N))
    max_entropy = np.log2(len(prob_dist))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    return normalized_entropy


def spectral_flux(sig, fs, frame_length=0.03, hop_length=0.015):
    """
    Vypočítá spektrální flux - míru změny spektra v čase.
    
    Týden 10-11: "Časově-frekvenční analýza"
    
    Flux měří, jak rychle se mění spektrum mezi sousedními okny.
    
    Vysoký flux = dynamický signál (rychlé změny)
    Nízký flux = stabilní signál (pomalé změny)
    
    Patologické hlasy mají často vyšší flux (nestabilita).
    
    Args:
        sig: Vstupní signál
        fs: Vzorkovací frekvence
        frame_length: Délka okna v sekundách
        hop_length: Posun okna v sekundách
    
    Returns:
        mean_flux: Průměrný spektrální flux
    """
    frame_samples = int(frame_length * fs)
    hop_samples = int(hop_length * fs)
    
    flux_values = []
    prev_spectrum = None
    
    # Sliding window přes signál
    for start in range(0, len(sig) - frame_samples, hop_samples):
        frame = sig[start:start + frame_samples]
        
        # Spektrum tohoto okna
        _, magnitude = get_frequency_spectrum(frame, fs)
        
        if prev_spectrum is not None:
            # Flux = suma čtvercových rozdílů mezi sousedními spektry
            flux = np.sum((magnitude - prev_spectrum) ** 2)
            flux_values.append(flux)
        
        prev_spectrum = magnitude
    
    if len(flux_values) == 0:
        return 0.0
    
    # Průměrný flux
    mean_flux = np.mean(flux_values)
    
    return mean_flux


def spectral_contrast(sig, fs, n_bands=6):
    """
    Vypočítá spektrální kontrast - rozdíl mezi píky a údolími ve spektru.
    
    Týden 10-11: "Fourierova analýza - harmonická struktura"
    
    Contrast měří, jak výrazné jsou harmonické proti šumu.
    
    Vysoký kontrast = jasně oddělené harmonické (zdravý hlas)
    Nízký kontrast = harmonické se mísí s šumem (patologický hlas)
    
    Args:
        sig: Vstupní signál
        fs: Vzorkovací frekvence
        n_bands: Počet frekvenčních pásem (default 6)
    
    Returns:
        mean_contrast: Průměrný spektrální kontrast přes všechna pásma
    """
    freqs, magnitude = get_frequency_spectrum(sig, fs)
    
    # Rozdělíme spektrum na logaritmicky rozmístěná pásma
    # (nízké frekvence jemnější rozdělení, vysoké hrubší)
    freq_bands = np.logspace(np.log10(freqs[1]), np.log10(freqs[-1]), n_bands + 1)
    
    contrasts = []
    
    for i in range(n_bands):
        # Najdeme indexy pro toto pásmo
        band_mask = (freqs >= freq_bands[i]) & (freqs < freq_bands[i + 1])
        band_magnitude = magnitude[band_mask]
        
        if len(band_magnitude) == 0:
            continue
        
        # Vrcholy (peaks) vs údolí (valleys)
        # Použijeme percentily: 90% percentil = píky, 10% percentil = údolí
        peak_value = np.percentile(band_magnitude, 90)
        valley_value = np.percentile(band_magnitude, 10)
        
        # Kontrast v tomto pásmu
        if valley_value > 0:
            contrast = peak_value / (valley_value + 1e-10)
        else:
            contrast = 0.0
        
        contrasts.append(contrast)
    
    if len(contrasts) == 0:
        return 0.0
    
    # Průměrný kontrast přes všechna pásma
    mean_contrast = np.mean(contrasts)
    
    return mean_contrast


def spectral_slope(sig, fs):
    """
    Vypočítá spektrální slope - sklon lineární regrese magnitudy spektra.
    
    Týden 10-11: "Frekvenční charakteristika"
    
    Slope měří, jak rychle klesá energie s frekvencí.
    
    Negativní slope = energie klesá s frekvencí (normální)
    Méně negativní slope = více vysokofrekvenční energie (může být šum)
    
    Args:
        sig: Vstupní signál
        fs: Vzorkovací frekvence
    
    Returns:
        slope: Spektrální slope
    """
    freqs, magnitude = get_frequency_spectrum(sig, fs)
    
    # Logaritmujeme obě osy (log-log plot)
    log_freqs = np.log10(freqs[1:] + 1)  # +1 pro zabránění log(0)
    log_magnitude = np.log10(magnitude[1:] + 1e-10)
    
    # Lineární regrese
    slope, _ = np.polyfit(log_freqs, log_magnitude, 1)
    
    return slope