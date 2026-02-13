import matplotlib.pyplot as plt
import numpy as np

def plot_time_signal(t, sig, title="Signál v čase", xlabel="Čas [s]", ylabel="Amplituda"):
    plt.figure(figsize=(10, 4))
    plt.plot(t, sig)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def plot_spectrum(freqs, magnitude, title="Frekvenční spektrum"):
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, magnitude)
    plt.title(title)
    plt.xlabel("Frekvence [Hz]")
    plt.ylabel("Magnituda")
    plt.grid(True)
    plt.show()

def compare_signals(t, sig1, sig2, label1="Původní", label2="Upravený"):
    plt.figure(figsize=(12, 5))
    plt.plot(t, sig1, label=label1, alpha=0.7)
    plt.plot(t, sig2, label=label2, alpha=0.7, linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()