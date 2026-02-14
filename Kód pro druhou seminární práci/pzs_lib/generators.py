import numpy as np
import scipy.signal as signal

def generate_time_vector(duration, fs):
    """Vytvoří časový vektor.
    duration: délka v sekundách
    fs: vzorkovací frekvence (Hz)
    """
    return np.linspace(0, duration, int(duration * fs), endpoint=False)

def sine_wave(t, f, amplitude=1.0, phase=0):
    """Generuje sinusovku: A * sin(2*pi*f*t + phi)"""
    return amplitude * np.sin(2 * np.pi * f * t + phase)

def cosine_wave(t, f, amplitude=1.0, phase=0):
    """Generuje kosinusovku (pro Fourierovy řady): A * cos(2*pi*f*t + phi)"""
    return amplitude * np.cos(2 * np.pi * f * t + phase)

def square_wave(t, f, amplitude=1.0, duty=0.5):
    """Generuje obdélníkový signál."""
    return amplitude * signal.square(2 * np.pi * f * t, duty=duty)

def sawtooth_wave(t, f, amplitude=1.0, width=1.0):
    """Generuje pilový (width=1) nebo trojúhelníkový (width=0.5) signál."""
    return amplitude * signal.sawtooth(2 * np.pi * f * t, width=width)

def add_noise(sig, snr_db=None, amplitude=None):
    """Přidá do signálu šum.
    Buď podle amplitudy (jak bylo v 'KumulaceTechniky.ipynb')
    nebo podle SNR (Signal-to-Noise Ratio).
    """
    noise = np.random.rand(len(sig)) - 0.5 # Centr na 0
    
    if amplitude is not None:
        return sig + amplitude * 2 * noise # *2 aby to bylo v rozsahu -amp až +amp
        
    # Pokročilejší přidání šumu přes SNR (pokud bys potřeboval)
    if snr_db is not None:
        sig_power = np.mean(sig ** 2)
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(sig))
        return sig + noise
        
    return sig