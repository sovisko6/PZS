import numpy as np
import scipy.signal as signal

def design_fir_filter(fs, cutoff_freq, num_taps=None, ripple_db=60.0, width_hz=None):
    """
    Navrhne koeficienty FIR filtru (typ dolní propust).
    Pokud není zadáno num_taps, vypočítá se automaticky pomocí Kaiserova vzorce.
    
    Args:
        fs: Vzorkovací frekvence
        cutoff_freq: Mezní frekvence (Hz)
        ripple_db: Útlum v nepropustném pásmu (dB) - default 60dB
        width_hz: Šířka přechodového pásma (Hz)
    """
    nyq_rate = fs / 2.0
    
    # Pokud není zadán počet tapů, odhadneme ho (jako v notebooku)
    if num_taps is None:
        if width_hz is None:
            width_hz = 5.0 # Defaultní šířka přechodu
        width = width_hz / nyq_rate
        # Funkce kaiserord vrátí potřebný řád filtru a beta parametr
        num_taps, beta = signal.kaiserord(ripple_db, width)
        # num_taps musí být liché pro symetrický FIR filtr typu I
        if num_taps % 2 == 0:
            num_taps += 1
    else:
        beta = 0.1102 * (ripple_db - 8.7) if ripple_db > 50 else 0.5842 * (ripple_db - 21) ** 0.4 + 0.07886 * (ripple_db - 21)
    
    # Výpočet koeficientů
    taps = signal.firwin(num_taps, cutoff_freq/nyq_rate, window=('kaiser', beta))
    return taps

def apply_filter(sig, taps):
    """Aplikuje FIR filtr na signál pomocí konvoluce (lfilter)."""
    # a=1.0 protože u FIR filtru jmenovatel přenosové funkce neexistuje
    return signal.lfilter(taps, 1.0, sig)

def moving_average(sig, window_size=5):
    """Jednoduchý filtr klouzavého průměru."""
    window = np.ones(window_size) / window_size
    return np.convolve(sig, window, mode='same')


def design_iir_filter(filter_type, fs, cutoff, order=4, btype='lowpass'):
    """
    Navrhne koeficienty IIR filtru (Butterworth, Chebyshev, Bessel).
    
    Týden 9: "Metody filtrace signálu"
    
    Args:
        filter_type: 'butter', 'cheby1', 'cheby2', 'bessel'
        fs: Vzorkovací frekvence
        cutoff: Mezní frekvence (nebo [low, high] pro bandpass)
        order: Řád filtru
        btype: 'lowpass', 'highpass', 'bandpass', 'bandstop'
    
    Returns:
        b, a: Koeficienty čitatele a jmenovatele
    """
    nyq = fs / 2.0
    
    if isinstance(cutoff, (list, tuple)):
        # Bandpass nebo bandstop
        normal_cutoff = [c / nyq for c in cutoff]
    else:
        normal_cutoff = cutoff / nyq
    
    if filter_type == 'butter':
        b, a = signal.butter(order, normal_cutoff, btype=btype)
    elif filter_type == 'cheby1':
        # Chebyshev Type I (ripple v propustném pásmu)
        b, a = signal.cheby1(order, 0.5, normal_cutoff, btype=btype)  # 0.5 dB ripple
    elif filter_type == 'cheby2':
        # Chebyshev Type II (ripple v nepropustném pásmu)
        b, a = signal.cheby2(order, 40, normal_cutoff, btype=btype)  # 40 dB útlum
    elif filter_type == 'bessel':
        b, a = signal.bessel(order, normal_cutoff, btype=btype)
    else:
        raise ValueError(f"Neznámý typ filtru: {filter_type}")
    
    return b, a


def apply_iir_filter(sig, b, a):
    """
    Aplikuje IIR filtr na signál.
    
    Args:
        sig: Vstupní signál
        b, a: Koeficienty filtru (z design_iir_filter)
    """
    return signal.filtfilt(b, a, sig)  # filtfilt = zero-phase filtering


def bandpass_filter(sig, fs, lowcut, highcut, order=4):
    """
    Pásmová propust (bandpass) - propouští frekvence mezi lowcut a highcut.
    
    Týden 9: "Metody filtrace signálu"
    
    Užitečné pro izolaci určitého frekvenčního pásma (např. lidský hlas 300-3400 Hz).
    """
    b, a = design_iir_filter('butter', fs, [lowcut, highcut], order=order, btype='bandpass')
    return apply_iir_filter(sig, b, a)


def highpass_filter(sig, fs, cutoff, order=4):
    """
    Horní propust (highpass) - propouští frekvence nad cutoff.
    
    Týden 9: "Metody filtrace signálu"
    
    Užitečné pro odstranění DC složky a nízkofrekvenčního driftu.
    """
    b, a = design_iir_filter('butter', fs, cutoff, order=order, btype='highpass')
    return apply_iir_filter(sig, b, a)


def notch_filter(sig, fs, freq_to_remove, Q=30):
    """
    Notch filter (pásmová zádrž) - odstraní úzké frekvenční pásmo.
    
    Týden 9: "Metody filtrace signálu"
    
    Typicky používáno pro odstranění 50/60 Hz síťového brumu.
    
    Args:
        freq_to_remove: Frekvence k odstranění (např. 50 Hz)
        Q: Kvalitní faktor (vyšší = užší zářez)
    """
    b, a = signal.iirnotch(freq_to_remove / (fs/2), Q)
    return apply_iir_filter(sig, b, a)


def adaptive_filter_lms(signal_input, desired, mu=0.01, filter_order=32):
    """
    Adaptivní filtr - LMS (Least Mean Squares) algoritmus.
    
    Týden 9: "Metody filtrace signálu (lineární filtry, adaptivní filtry aj.)"
    
    Adaptivní filtr se "učí" minimalizovat chybu mezi výstupem a požadovaným signálem.
    
    Args:
        signal_input: Vstupní signál (referenční)
        desired: Požadovaný výstup
        mu: Learning rate (krok adaptace)
        filter_order: Počet koeficientů filtru
    
    Returns:
        output: Výstup filtru
        error: Chybový signál
        weights: Finální váhy filtru
    """
    n = len(signal_input)
    weights = np.zeros(filter_order)
    output = np.zeros(n)
    error = np.zeros(n)
    
    for i in range(filter_order, n):
        # Vektor vstupních vzorků
        x = signal_input[i-filter_order:i][::-1]
        
        # Výstup filtru (konvoluce)
        output[i] = np.dot(weights, x)
        
        # Chyba
        error[i] = desired[i] - output[i]
        
        # Aktualizace vah (LMS)
        weights = weights + mu * error[i] * x
    
    return output, error, weights