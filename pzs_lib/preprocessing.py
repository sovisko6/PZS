"""
Preprocessing modul pro hlasové signály.

Týden 4: Segmentace a windowing
Týden 7-8: Filtrace signálů
"""

import numpy as np
from scipy.signal import butter, filtfilt, get_window


def voice_activity_detection(sig, fs, frame_length=0.03, hop_length=0.015, 
                             energy_threshold=0.001, zcr_threshold=0.3):
    """
    Voice Activity Detection (VAD) - detekuje segmenty s aktivním hlasem.
    
    Týden 4: "Segmentace signálu a windowing"
    
    VAD odstraní tiché části na začátku/konci signálu a pauzy uvnitř.
    Důležité pro extrakci příznaků (nechceme analyzovat ticho!).
    
    Používá dvě metriky:
    - Krátkodobou energii (hlasitost)
    - Zero-Crossing Rate (frekvence změn)
    
    Args:
        sig: Vstupní signál
        fs: Vzorkovací frekvence
        frame_length: Délka okna v sekundách (default 30ms)
        hop_length: Posun okna v sekundách (default 15ms)
        energy_threshold: Práh pro energii (0-1, default 0.001 - VELMI NÍZKÝ!)
        zcr_threshold: Práh pro ZCR (0-1, default 0.3 - VYSOKÝ!)
    
    Returns:
        active_signal: Signál obsahující jen aktivní segmenty
        voice_mask: Boolean maska (True = hlas, False = ticho)
    """
    frame_samples = int(frame_length * fs)
    hop_samples = int(hop_length * fs)
    
    # Normalizace signálu
    sig_norm = sig / (np.max(np.abs(sig)) + 1e-10)
    
    num_frames = (len(sig_norm) - frame_samples) // hop_samples + 1
    voice_mask = np.zeros(len(sig_norm), dtype=bool)
    
    # Sliding window analýza
    for i in range(num_frames):
        start = i * hop_samples
        end = start + frame_samples
        
        if end > len(sig_norm):
            break
        
        frame = sig_norm[start:end]
        
        # 1. Krátkodobá energie
        energy = np.sum(frame ** 2) / len(frame)
        
        # 2. Zero-Crossing Rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
        
        # 3. Rozhodnutí: Hlas pokud má dostatečnou energii A rozumný ZCR
        # (příliš vysoký ZCR = šum, příliš nízký = DC offset)
        if energy > energy_threshold and zero_crossings < zcr_threshold:
            voice_mask[start:end] = True
    
    # BEZPEČNOSTNÍ OPATŘENÍ: Pokud VAD odfiltroval VŠECHNO, vrať původní signál
    active_signal = sig[voice_mask]
    
    if len(active_signal) < 1000:  # Méně než 1000 vzorků = pravděpodobně chyba
        print(f"⚠️  VAD warning: Příliš málo dat po VAD ({len(active_signal)} vzorků), vracím původní signál")
        return sig, np.ones(len(sig), dtype=bool)
    else:
        print(f"✅ VAD info: Původní délka signálu {len(sig)} vzorků, po VAD {len(active_signal)} vzorků")
    return active_signal, voice_mask


def pre_emphasis(sig, alpha=0.97):
    """
    Pre-emphasis filtr - zvýrazní vyšší frekvence.
    
    Týden 7-8: "Filtrace signálů"
    
    Lidský hlas přirozeně klesá s frekvencí (~6 dB/oktáva).
    Pre-emphasis to kompenzuje a zvýrazní formanty (důležité pro rozpoznávání).
    
    Matematika: y[n] = x[n] - α * x[n-1]
    
    Args:
        sig: Vstupní signál
        alpha: Koeficient pre-emphasis (typicky 0.95-0.98)
    
    Returns:
        emphasized_sig: Signál s pre-emphasis
    """
    # High-pass FIR filtr 1. řádu
    emphasized_sig = np.append(sig[0], sig[1:] - alpha * sig[:-1])
    
    return emphasized_sig


def de_emphasis(sig, alpha=0.97):
    """
    De-emphasis filtr - inverzní operace k pre_emphasis.
    
    Týden 7-8: "Filtrace signálů"
    
    Args:
        sig: Pre-emphasized signál
        alpha: Koeficient (musí být stejný jako při pre-emphasis)
    
    Returns:
        original_sig: Původní signál
    """
    # Inverzní filtr: y[n] = x[n] + α * y[n-1]
    deemphasized_sig = np.zeros_like(sig)
    deemphasized_sig[0] = sig[0]
    
    for n in range(1, len(sig)):
        deemphasized_sig[n] = sig[n] + alpha * deemphasized_sig[n - 1]
    
    return deemphasized_sig


def bandpass_filter(sig, fs, lowcut=80, highcut=8000, order=5):
    """
    Band-pass Butterworth filtr - propustí jen frekvenční pásmo hlasu.
    
    Týden 7-8: "Návrh a aplikace digitálních filtrů"
    
    Lidský hlas je typicky v rozsahu 80-8000 Hz.
    Toto odfiltruje:
    - DC offset a velmi nízké frekvence (<80 Hz)
    - Vysokofrekvenční šum (>8000 Hz)
    - Síťový brum (50 Hz)
    
    Args:
        sig: Vstupní signál
        fs: Vzorkovací frekvence
        lowcut: Dolní mezní frekvence [Hz] (default 80 Hz)
        highcut: Horní mezní frekvence [Hz] (default 8000 Hz)
        order: Řád filtru (default 5)
    
    Returns:
        filtered_sig: Filtrovaný signál
    """
    # Návrh Butterworth band-pass filtru
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Ošetření okrajových případů
    if low <= 0:
        low = 0.01
    if high >= 1:
        high = 0.99
    
    b, a = butter(order, [low, high], btype='band')
    
    # Aplikace filtru (filtfilt = zero-phase filtering)
    filtered_sig = filtfilt(b, a, sig)
    
    return filtered_sig


def notch_filter(sig, fs, freq=50, Q=30):
    """
    Notch filtr - odstraní úzké frekvenční pásmo (např. síťový brum).
    
    Týden 7-8: "Návrh a aplikace digitálních filtrů"
    
    Síťový brum (50 Hz v EU, 60 Hz v USA) je častý problém v audio nahrávkách.
    
    Args:
        sig: Vstupní signál
        fs: Vzorkovací frekvence
        freq: Frekvence k odstranění [Hz] (default 50 Hz)
        Q: Kvalita filtru (užší = vyšší Q) (default 30)
    
    Returns:
        filtered_sig: Filtrovaný signál
    """
    from scipy.signal import iirnotch
    
    # Návrh notch filtru
    nyquist = fs / 2
    w0 = freq / nyquist
    
    b, a = iirnotch(w0, Q)
    
    # Aplikace filtru
    filtered_sig = filtfilt(b, a, sig)
    
    return filtered_sig


def apply_window(sig, window_type='hamming'):
    """
    Aplikuje okénkovací funkci na signál.
    
    Týden 4: "Windowing - redukce spektrálního úniku"
    
    Okénkování je důležité před FFT pro redukci spektrálního úniku (leakage).
    
    Typy oken:
    - 'hamming': Dobrá kompromis mezi únikem a rozlišením (nejčastější)
    - 'hann': Podobné hamming, o trochu lepší potlačení úniků
    - 'blackman': Velmi dobré potlačení úniků, ale nižší rozlišení
    - 'kaiser': Nastavitelný trade-off (potřebuje beta parametr)
    
    Args:
        sig: Vstupní signál
        window_type: Typ okna ('hamming', 'hann', 'blackman', 'kaiser')
    
    Returns:
        windowed_sig: Signál s aplikovaným oknem
    """
    window = get_window(window_type, len(sig))
    windowed_sig = sig * window
    
    return windowed_sig


def segment_signal(sig, fs, frame_length=0.03, hop_length=0.015, window_type='hamming'):
    """
    Rozdělí signál na překrývající se segmenty s okny.
    
    Týden 4: "Segmentace a overlap processing"
    
    Používá se pro:
    - Short-Time Fourier Transform (STFT)
    - Krátkodobou analýzu energie, ZCR atd.
    
    Args:
        sig: Vstupní signál
        fs: Vzorkovací frekvence
        frame_length: Délka segmentu v sekundách (default 30ms)
        hop_length: Posun mezi segmenty v sekundách (default 15ms = 50% overlap)
        window_type: Typ okénkovací funkce
    
    Returns:
        segments: 2D pole (num_frames, frame_samples)
    """
    frame_samples = int(frame_length * fs)
    hop_samples = int(hop_length * fs)
    
    # Okénkovací funkce
    window = get_window(window_type, frame_samples)
    
    segments = []
    
    # Sliding window
    for start in range(0, len(sig) - frame_samples, hop_samples):
        segment = sig[start:start + frame_samples]
        windowed_segment = segment * window
        segments.append(windowed_segment)
    
    return np.array(segments)


def preprocess_voice_complete(sig, fs, apply_vad=True, apply_preemph=True, 
                               apply_bandpass=True, apply_notch=True):
    """
    Kompletní preprocessing pipeline pro hlasový signál.
    
    Kombinuje všechny preprocessing kroky v optimálním pořadí:
    1. Band-pass filter (odstranění DC a vysokofrekvenčního šumu)
    2. Notch filter (odstranění síťového brumu)
    3. Voice Activity Detection (odstranění ticha)
    4. Pre-emphasis (zvýraznění vysokých frekvencí)
    
    Args:
        sig: Vstupní signál
        fs: Vzorkovací frekvence
        apply_vad: Aplikovat VAD (default True)
        apply_preemph: Aplikovat pre-emphasis (default True)
        apply_bandpass: Aplikovat band-pass filter (default True)
        apply_notch: Aplikovat notch filter (default True)
    
    Returns:
        processed_sig: Preprocessovaný signál
        metadata: Dictionary s informacemi o preprocessing
    """
    metadata = {
        'original_length': len(sig),
        'steps_applied': []
    }
    
    processed_sig = sig.copy()
    
    # Krok 1: Band-pass filter (80-8000 Hz)
    if apply_bandpass:
        processed_sig = bandpass_filter(processed_sig, fs)
        metadata['steps_applied'].append('bandpass_filter')
    
    # Krok 2: Notch filter (50 Hz)
    if apply_notch:
        processed_sig = notch_filter(processed_sig, fs, freq=50)
        metadata['steps_applied'].append('notch_filter')
    
    # Krok 3: Voice Activity Detection
    if apply_vad:
        processed_sig, voice_mask = voice_activity_detection(processed_sig, fs)
        metadata['steps_applied'].append('vad')
        metadata['vad_retention'] = len(processed_sig) / len(sig)
    
    # Krok 4: Pre-emphasis
    if apply_preemph:
        processed_sig = pre_emphasis(processed_sig)
        metadata['steps_applied'].append('pre_emphasis')
    
    metadata['processed_length'] = len(processed_sig)
    
    return processed_sig, metadata
