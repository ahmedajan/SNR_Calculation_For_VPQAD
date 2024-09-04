# %%
import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d

def calculate_segmental_snr_high(audio_signal, base_sigma=1.0, segment_length=1000, adaptive_factor=0.005, smoothing_weight=5):
    """
    Calculating the Segmental SNR (SegSNR) of a given speech signal with more aggressive noise suppression
    and weighted segments for higher SNR values.
    
    Args:
    audio_signal (numpy array): Input audio signal.
    base_sigma (float): Base standard deviation for Gaussian smoothing (default is 1.0).
    segment_length (int): Number of samples per segment (default is 1000).
    adaptive_factor (float): Factor to detect noise-heavy regions adaptively (default is 0.005).
    smoothing_weight (float): Weight factor to increase smoothing in noise-dominant regions (default is 5).
    
    Returns:
    seg_snr (float): Weighted average Segmental SNR (SegSNR) in dB.
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

    # To store SNR for each segment and their weights
    segment_snr_list = []
    segment_weights = []

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

        # Calculating segment-wise power for signal and noise
        power_signal = np.mean(segment ** 2)
        power_noise = np.mean(noise ** 2)

        # Calculating SNR for this segment in dB
        segment_snr = 10 * np.log10(power_signal / power_noise)
        segment_snr_list.append(segment_snr)

        # Giving higher weight to signal-dominant segments
        weight = power_signal / (power_signal + power_noise)
        segment_weights.append(weight)

    # Calculating weighted average SegSNR across all segments
    seg_snr = np.average(segment_snr_list, weights=segment_weights)

    return seg_snr

# Importing the audio file
file_path = 'P01_S01_TD.wav'  # Replacing with your file path
audio_signal, fs = sf.read(file_path)  # Reading audio data and sampling rate

# Calculating the Segmental SNR from the imported audio signal with more aggressive smoothing and weights
seg_snr_estimated = calculate_segmental_snr_high(audio_signal, base_sigma=1.0, segment_length=1000, adaptive_factor=0.005, smoothing_weight=5)

# Outputting the estimated Segmental SNR
print(f"Segmented SNR (SegSNR): {seg_snr_estimated:.2f} dB")




