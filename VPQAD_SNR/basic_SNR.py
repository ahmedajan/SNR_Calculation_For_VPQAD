# %%
import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d

def estimate_snr_weighted(audio_signal, base_sigma=1.2, min_segment_length=500, max_segment_length=1500, adaptive_factor=0.01):
    """
    Estimating the Signal-to-Noise Ratio (SNR) of a given speech signal using weighted adaptive smoothing,
    dynamic segment lengths, and signal power scaling for higher SNR.
    
    Args:
    audio_signal (numpy array): Input audio signal.
    base_sigma (float): Base standard deviation for Gaussian smoothing (default is 1.2).
    min_segment_length (int): Minimum number of samples per segment for dynamic smoothing (default is 500).
    max_segment_length (int): Maximum number of samples per segment for dynamic smoothing (default is 1500).
    adaptive_factor (float): Factor to help detect noise-heavy regions adaptively (default is 0.01).
    
    Returns:
    snr (float): Estimated SNR in dB.
    """

    # Checking if the audio has multiple channels (e.g., stereo), and converting to mono if necessary
    if len(audio_signal.shape) > 1:
        audio_signal = np.mean(audio_signal, axis=1)

    # Normalizing the audio signal
    normalized_signal = audio_signal / np.max(np.abs(audio_signal))

    # Step 1: Calculating the power of the signal (normalized signal)
    power_signal = np.mean(normalized_signal ** 2)

    # Step 2: Setting adaptive noise threshold based on signal energy
    max_signal_value = np.max(np.abs(normalized_signal))
    adaptive_noise_threshold = adaptive_factor * max_signal_value  # Setting adaptive threshold based on signal's energy

    # Step 3: Applying weighted smoothing with dynamic segment length
    smoothed_signal = np.zeros_like(normalized_signal)
    total_samples = len(normalized_signal)
    pos = 0

    while pos < total_samples:
        # Dynamically adjusting segment length based on signal energy
        segment_length = np.random.randint(min_segment_length, max_segment_length)
        end = min(pos + segment_length, total_samples)

        # Segmenting the signal
        segment = normalized_signal[pos:end]
        segment_power = np.abs(segment).mean()

        # Dynamically adjusting sigma based on segment power and applying weighted smoothing
        if segment_power < adaptive_noise_threshold:
            # Applying heavier smoothing for noise-dominant regions (weighted)
            local_sigma = base_sigma + 3  # Applying more aggressive smoothing for noise
        else:
            # Applying less smoothing for signal-dominant regions
            local_sigma = base_sigma

        # Applying smoothing to the segment
        smoothed_signal[pos:end] = gaussian_filter1d(segment, sigma=local_sigma)

        # Moving to the next segment
        pos = end

    # Step 4: Estimating the noise by subtracting the smoothed signal from the original
    noise = normalized_signal - smoothed_signal

    # Step 5: Calculating the power of the noise
    power_noise = np.mean(noise ** 2)

    # Step 6: Computing the SNR in dB
    snr = 10 * np.log10(power_signal / power_noise)

    return snr

# Importing the audio file
file_path = 'P01_S01_TD.wav'  # Replacing with your file path
audio_signal, fs = sf.read(file_path)  # Reading audio data and sampling rate

# Estimating the SNR from the imported audio signal with weighted adaptive smoothing
snr_estimated = estimate_snr_weighted(audio_signal, base_sigma=1.2, min_segment_length=500, max_segment_length=1500, adaptive_factor=0.01)

# Outputting the estimated SNR
print(f"SNR: {snr_estimated:.2f} dB")




