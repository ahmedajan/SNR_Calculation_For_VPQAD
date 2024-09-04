# %%
import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
from scipy.signal import get_window
from scipy.fftpack import fft

def calculate_frequency_weighted_snr(audio_signal, base_sigma=1.0, segment_length=1000, adaptive_factor=0.005, smoothing_weight=5, weighting_function='A'):
    """
    Calculating the Frequency-Weighted SNR (FWSNR) of a given speech signal by applying a weighting function 
    to the frequency components of the signal and noise.
    
    Args:
    audio_signal (numpy array): Input audio signal.
    base_sigma (float): Base standard deviation for Gaussian smoothing (default is 1.0).
    segment_length (int): Number of samples per segment (default is 1000).
    adaptive_factor (float): Factor to detect noise-heavy regions adaptively (default is 0.005).
    smoothing_weight (float): Weight factor to increase smoothing in noise-dominant regions (default is 5).
    weighting_function (str): Weighting function to use (default is 'A' for A-weighting).
    
    Returns:
    fw_snr (float): Frequency-Weighted SNR (FWSNR) in dB (always positive).
    """

    # Checking if the audio has multiple channels (e.g., stereo), and converting to mono if necessary
    if len(audio_signal.shape) > 1:
        audio_signal = np.mean(audio_signal, axis=1)

    # Normalizing the audio signal
    normalized_signal = audio_signal / np.max(np.abs(audio_signal))

    # Calculating the maximum signal value for adaptive noise thresholding
    max_signal_value = np.max(np.abs(normalized_signal))
    adaptive_noise_threshold = adaptive_factor * max_signal_value  # Setting adaptive threshold

    total_samples = len(normalized_signal)
    num_segments = total_samples // segment_length

    # To store weighted SNR for each frequency band
    frequency_snr_list = []

    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length

        # Segmenting the signal
        segment = normalized_signal[start:end]
        segment_power = np.abs(segment).mean()

        # Dynamically adjusting sigma for noise-heavy and signal-dominant segments
        if segment_power < adaptive_noise_threshold:
            local_sigma = base_sigma + smoothing_weight  # Heavier smoothing for noise-dominant regions
        else:
            local_sigma = base_sigma  # Less smoothing for signal-dominant regions

        # Smoothing the segment
        smoothed_segment = gaussian_filter1d(segment, sigma=local_sigma)

        # Estimating noise for the segment
        noise = segment - smoothed_segment

        # Applying window function for FFT (Hann window)
        window = get_window('hann', segment_length)

        # Compute FFT for both signal and noise
        fft_signal = np.abs(fft(segment * window))
        fft_noise = np.abs(fft(noise * window))

        # Apply frequency weighting (e.g., A-weighting)
        if weighting_function == 'A':
            freq_weight = apply_a_weighting(len(fft_signal), fs)

        # Calculating SNR for each frequency bin
        freq_snr = 10 * np.log10(np.maximum(fft_signal ** 2, 1e-12) / np.maximum(fft_noise ** 2, 1e-12))  # Avoid divide by zero and log of zero

        # Apply the weighting to the SNR of each frequency bin
        weighted_snr = np.mean(freq_snr * freq_weight)
        frequency_snr_list.append(weighted_snr)

    # Averaging across all segments to compute the final Frequency-Weighted SNR
    fw_snr = np.mean(frequency_snr_list)

    # Ensure the final result is positive by taking the absolute value
    return np.abs(fw_snr)

def apply_a_weighting(num_freqs, fs):
    """
    Applying A-weighting to the frequency components.
    A-weighting emphasizes frequencies in the range of human speech (500 Hz to 6 kHz).
    
    Args:
    num_freqs (int): Number of frequency bins.
    fs (int): Sampling rate.
    
    Returns:
    freq_weight (numpy array): A-weighting applied to the frequency bins.
    """
    # Generate frequencies
    freqs = np.linspace(0, fs / 2, num_freqs)

    # A-weighting formula coefficients (approximation)
    ra = ((12200 ** 2) * (freqs ** 4)) / ((freqs ** 2 + 20.6 ** 2) * np.sqrt((freqs ** 2 + 107.7 ** 2) * (freqs ** 2 + 737.9 ** 2)) * (freqs ** 2 + 12200 ** 2))
    
    # Convert to dB scale
    a_weighting = 20 * np.log10(np.maximum(ra, 1e-12)) + 2.00  # Add correction factor and avoid log of zero

    # Normalize to prevent extreme values
    a_weighting = np.maximum(a_weighting, -80)

    return np.interp(np.linspace(0, fs / 2, num_freqs), freqs, a_weighting)

# Importing the audio file
file_path = 'P01_S01_TD.wav'  # Replace with your file path
audio_signal, fs = sf.read(file_path)  # Reading audio data and sampling rate

# Calculating the Frequency-Weighted SNR from the imported audio signal
fw_snr_estimated = calculate_frequency_weighted_snr(audio_signal, base_sigma=1.0, segment_length=1000, adaptive_factor=0.005, smoothing_weight=5)

# Outputting the estimated Frequency-Weighted SNR
print(f"Frequency-Weighted SNR (FWSNR): {fw_snr_estimated:.2f} dB")



