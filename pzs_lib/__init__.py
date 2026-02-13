from .generators import generate_time_vector, sine_wave, square_wave, sawtooth_wave, add_noise
from .time_analysis import (calculate_energy, statistics, autocorrelation, custom_convolution, 
                            find_peaks, resample_signal, normalize_signal, covariance, cross_correlation,
                            ensemble_averaging, moving_average_cumulative, correlation_peak_detection, matched_filter,
                            calculate_hnr, calculate_jitter, calculate_shimmer, calculate_zcr, 
                            calculate_energy_variability)
from .freq_analysis import (dft_slow, get_frequency_spectrum, compute_real_cepstrum, analyze_voice_features,
                            spectral_centroid, spectral_flatness, spectral_rolloff, spectral_bandwidth,
                            hilbert_transform, instantaneous_frequency, spectral_entropy, spectral_flux,
                            spectral_contrast, spectral_slope, analyze_quefrency_width)
from .filters import (design_fir_filter, apply_filter, moving_average,
                     design_iir_filter, apply_iir_filter, bandpass_filter, highpass_filter,
                     notch_filter, adaptive_filter_lms)
from .preprocessing import (voice_activity_detection, pre_emphasis, de_emphasis,
                            bandpass_filter as preprocessing_bandpass, notch_filter as preprocessing_notch,
                            apply_window, segment_signal, preprocess_voice_complete)
from .visualization import plot_time_signal, plot_spectrum, compare_signals