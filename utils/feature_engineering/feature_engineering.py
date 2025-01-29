import numpy as np
import pandas as pd
from scipy.stats import zscore, skew, kurtosis
import librosa
import stumpy
import warnings

def compute_matrix_profile(wave, window_size):

    matrix_profile = stumpy.stump(wave.squeeze().asdtype('float64'), m=window_size)

    return matrix_profile[:, 0], matrix_profile[:, 1]

def outliers(X):

    z_scores = np.abs(zscore(X))
    threshold = 3 # 3 standard deviations
    outliers = np.where(z_scores > threshold)

    return outliers

def drop_unused_columns(df):
    data = df.copy()
    data.drop(columns=['path'], inplace=True)
    data.drop(columns=['sampling_rate'], inplace=True)
    data = data.copy()
    return data

def encode_gender(df):
    
    data = df.copy()
    data = pd.get_dummies(data, columns=['gender'], prefix='gender')

    expected_genders = ['male', 'female']
    for gender in expected_genders:
        col_name = f'gender_{gender}'
        if col_name not in data.columns:
            data[col_name] = False  # Add missing column with all zeros

    data = data.copy()

    return data

def encode_ethnicity(df):

    data = df.copy()
    data['ethnicity_grouped'] = data['ethnicity'].apply(
        lambda x: x if x in ['igbo', 'english'] else 'others' # Map ethnicities to 'igbo', 'english', or 'others'
    )
    data = pd.get_dummies(data, columns=['ethnicity_grouped'], prefix='ethnicity')

    expected_ethnicities = ['igbo', 'english', 'others']
    for ethnicity in expected_ethnicities:
        col_name = f'ethnicity_{ethnicity}'
        if col_name not in data.columns:
            data[col_name] = False  # Add missing column with all zeros

    data.drop(columns=['ethnicity'], inplace=True)
    data = data.copy()
    
    return data

def floatize_tempo(df):
    data = df.copy()
    data['tempo'] = data['tempo'].apply(lambda x: float(x.strip('[]')))
    
    return data

def comb_precomp(df):
    data = df.copy()

    data['characters_per_word'] = (data['num_characters'] / data['num_words']).fillna(0)
    data['words_per_second'] = data['num_words'] / data['duration']
    data['pitch_range'] = data['max_pitch'] - data['min_pitch']
    data['mean_to_max_pitch_ratio'] = data['mean_pitch'] / data['max_pitch']
    data['energy_to_duration_ratio'] = data['energy'] / data['duration']
    data['energy_to_silence_ratio'] = data['energy'] / data['silence_duration']
    data['num_words_per_silence'] = data['num_words'] / data['silence_duration']
    data['silence_ratio'] = data['silence_duration'] / data['duration']

    data = data.copy()
    return data

def log_mel_spec(df, S=False, sr=22050, n_mels=20, n_fft=2048, hop_length=512):
    data = df.copy()

    # Compute log Mel spectrogram for each audio
    if S:
        log_melspectrogram = data['log_melspectrogram']
    else:
        log_melspectrogram = data['wave'].apply(
            lambda y: librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels))
        )

    # Calculate mean, median, and std for each Mel band
    log_spec_mean = log_melspectrogram.apply(lambda x: np.mean(x, axis=1))
    for i in range(n_mels):
        data[f'log_melspec_mean_{i}'] = log_spec_mean.apply(lambda x: x[i])

    log_spec_median = log_melspectrogram.apply(lambda x: np.median(x, axis=1))
    for i in range(n_mels):
        data[f'log_melspec_median_{i}'] = log_spec_median.apply(lambda x: x[i])

    log_spec_std = log_melspectrogram.apply(lambda x: np.std(x, axis=1))
    for i in range(n_mels):
        data[f'log_melspec_std_{i}'] = log_spec_std.apply(lambda x: x[i])

    # Compute global statistics directly on all Mel bands
    data['log_melspec_mean'] = log_melspectrogram.apply(lambda x: np.mean(x))
    data['log_melspec_median'] = log_melspectrogram.apply(lambda x: np.median(x))
    data['log_melspec_std'] = log_melspectrogram.apply(lambda x: np.std(x))
    data['log_melspec_skewness'] = log_melspectrogram.apply(lambda x: skew(x.flatten()))
    data['log_melspec_kurtosis'] = log_melspectrogram.apply(lambda x: kurtosis(x.flatten()))

    # Compute delta log Mel spectrogram for each audio
    delta_log_melspectrogram = log_melspectrogram.apply(lambda log_mel: librosa.feature.delta(log_mel))

    # Compute delta log Mel spectrogram global statistics for each Mel band
    delta_log_spec_means = delta_log_melspectrogram.apply(lambda delta: np.mean(delta, axis=1))
    delta_log_spec_medians = delta_log_melspectrogram.apply(lambda delta: np.median(delta, axis=1))
    delta_log_spec_stds = delta_log_melspectrogram.apply(lambda delta: np.std(delta, axis=1))

    warnings.filterwarnings("ignore")
    for i in range(n_mels):
        data[f'delta_log_melspec_mean_{i}'] = delta_log_spec_means.apply(lambda x: x[i])
        data[f'delta_log_melspec_median_{i}'] = delta_log_spec_medians.apply(lambda x: x[i])
        data[f'delta_log_melspec_std_{i}'] = delta_log_spec_stds.apply(lambda x: x[i])
    warnings.filterwarnings("default")

    data = data.copy()
    return data

def mfcc(df, S=False, sr=22050, n_mfcc=20, n_fft=2048, hop_length=512):
    data = df.copy()

    # Compute MFCC for each audio
    if S:
        mfcc = data['log_melspectrogram'].apply(
            lambda S: librosa.feature.mfcc(S=S, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        )
    else:
        mfcc = data['wave'].apply(
            lambda y: librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        )
    
    # Compute means and stds per MFCC coefficient
    mfcc_means = mfcc.apply(lambda mfcc: np.mean(mfcc, axis=1))
    mfcc_medians = mfcc.apply(lambda mfcc: np.median(mfcc, axis=1))
    mfcc_stds = mfcc.apply(lambda mfcc: np.std(mfcc, axis=1))

    for i in range(n_mfcc):
        data[f'mfcc_mean_{i}'] = mfcc_means.apply(lambda x: x[i])
    
    for i in range(n_mfcc):
        data[f'mfcc_median_{i}'] = mfcc_medians.apply(lambda x: x[i])

    for i in range(n_mfcc):
        data[f'mfcc_std_{i}'] = mfcc_stds.apply(lambda x: x[i])

    # Compute global statistics directly on all MFCC coefficients
    data['mfcc_mean'] = mfcc.apply(lambda mfcc: np.mean(mfcc))
    data['mfcc_median'] = mfcc.apply(lambda mfcc: np.median(mfcc))
    data['mfcc_std'] = mfcc.apply(lambda mfcc: np.std(mfcc))
    data['mfcc_skewness'] = mfcc.apply(lambda mfcc: skew(mfcc.flatten()))
    data['mfcc_kurtosis'] = mfcc.apply(lambda mfcc: kurtosis(mfcc.flatten()))

    # Compute delta MFCC for each audio
    delta_mfcc = mfcc.apply(lambda mfcc_matrix: librosa.feature.delta(mfcc_matrix))

    # Compute delta MFCC global statistics for each MFCC coefficient
    delta_mfcc_means = delta_mfcc.apply(lambda delta: np.mean(delta, axis=1))
    delta_mfcc_medians = delta_mfcc.apply(lambda delta: np.median(delta, axis=1))
    delta_mfcc_stds = delta_mfcc.apply(lambda delta: np.std(delta, axis=1))

    warnings.filterwarnings("ignore")
    for i in range(n_mfcc):
        data[f'delta_mfcc_mean_{i}'] = delta_mfcc_means.apply(lambda x: x[i])
        data[f'delta_mfcc_median_{i}'] = delta_mfcc_medians.apply(lambda x: x[i])
        data[f'delta_mfcc_std_{i}'] = delta_mfcc_stds.apply(lambda x: x[i])
    warnings.filterwarnings("default")

    data = data.copy()
    return data

def rms(df, S=False, frame_length=2048, hop_length=512):
    data = df.copy()

    # Compute RMS
    if S:
        rms_features = data['spectrogram'].apply(
            lambda S: librosa.feature.rms(S=S, frame_length=frame_length, hop_length=hop_length)
        )
    else:
        rms_features = data['wave'].apply(
            lambda y: librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
        )

    data['rms_mean'] = rms_features.apply(np.mean)
    data['rms_median'] = rms_features.apply(np.median)
    data['rms_std'] = rms_features.apply(np.std)
    #data['rms_skewness'] = rms_features.apply(skew) # TODO: Sometimes return nan
    #data['rms_kurtosis'] = rms_features.apply(kurtosis) # TODO: Sometimes return nan

    data = data.copy()
    return data

def silence_duration_contour(df, num_silence_frames=20, sr=22050):
    data = df.copy()

    # Extract silence durations once and calculate features
    silence_duration_frames = data['wave'].apply(
        lambda wave: extract_silence_durations(wave, sr=sr, num_frames=num_silence_frames)
    )

    frame_duration = (data['duration'] / num_silence_frames)
    for i in range(num_silence_frames):
        data[f'silence_duration_frame_{i}'] = silence_duration_frames.apply(lambda x: x[i])
        data[f'silence_ratio_on_frame_{i}'] = silence_duration_frames.apply(lambda x: ((x[i]) / frame_duration[i])) 


    # Global statistics
    data['silence_duration_frames_mean'] = silence_duration_frames.apply(np.mean)
    data['silence_duration_frames_median'] = silence_duration_frames.apply(np.median)
    data['silence_duration_frames_std'] = silence_duration_frames.apply(np.std)
    #data['silence_duration_frames_skewness'] = silence_duration_frames.apply(skew) # TODO: Sometimes return nan
    #data['silence_duration_frames_kurtosis'] = silence_duration_frames.apply(kurtosis) # TODO: Sometimes return nan

    data = data.copy()
    return data

def spectral(df, S=False, sr=22050, n_fft=2048, hop_length=512):
    data = df.copy()

    if S:
        # Compute spectral features from the spectrogram
        spectral_centroids = data['spectrogram'].apply(
            lambda S: librosa.feature.spectral_centroid(S=S, sr=sr, n_fft=n_fft, hop_length=hop_length)
        )
        spectral_bandwidths = data['spectrogram'].apply(
            lambda S: librosa.feature.spectral_bandwidth(S=S, sr=sr, n_fft=n_fft, hop_length=hop_length)
        )
        spectral_contrasts = data['spectrogram'].apply(
            lambda S: librosa.feature.spectral_contrast(S=S, sr=sr, n_fft=n_fft, hop_length=hop_length)
        )
        spectral_flatnesses = data['spectrogram'].apply(
            lambda S: librosa.feature.spectral_flatness(S=S, n_fft=n_fft, hop_length=hop_length)
        )
        spectral_rolloffs = data['spectrogram'].apply(
            lambda S: librosa.feature.spectral_rolloff(S=S, sr=sr, n_fft=n_fft, hop_length=hop_length)
        )
    else:
        # Compute spectral features from the waveforms
        spectral_centroids = data['wave'].apply(
            lambda y: librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        )
        spectral_bandwidths = data['wave'].apply(
            lambda y: librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        )
        spectral_contrasts = data['wave'].apply(
            lambda y: librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        )
        spectral_flatnesses = data['wave'].apply(
            lambda y: librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)
        )
        spectral_rolloffs = data['wave'].apply(
            lambda y: librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        )

    # Compute global statistics for each feature and add to the DataFrame
    # Spectral Centroid
    data['spectral_centroid_mean'] = spectral_centroids.apply(lambda x: np.mean(x))
    data['spectral_centroid_median'] = spectral_centroids.apply(lambda x: np.median(x))
    data['spectral_centroid_std'] = spectral_centroids.apply(lambda x: np.std(x))

    # Spectral Bandwidth
    data['spectral_bandwidth_mean'] = spectral_bandwidths.apply(lambda x: np.mean(x))
    data['spectral_bandwidth_median'] = spectral_bandwidths.apply(lambda x: np.median(x))
    data['spectral_bandwidth_std'] = spectral_bandwidths.apply(lambda x: np.std(x))

    # Spectral Contrast
    data['spectral_contrast_mean'] = spectral_contrasts.apply(lambda x: np.mean(x))
    data['spectral_contrast_median'] = spectral_contrasts.apply(lambda x: np.median(x))
    data['spectral_contrast_std'] = spectral_contrasts.apply(lambda x: np.std(x))

    # Spectral Flatness
    data['spectral_flatness_mean'] = spectral_flatnesses.apply(lambda x: np.mean(x))
    data['spectral_flatness_median'] = spectral_flatnesses.apply(lambda x: np.median(x))
    data['spectral_flatness_std'] = spectral_flatnesses.apply(lambda x: np.std(x))

    # Spectral Rolloff
    data['spectral_rolloff_mean'] = spectral_rolloffs.apply(lambda x: np.mean(x))
    data['spectral_rolloff_median'] = spectral_rolloffs.apply(lambda x: np.median(x))
    data['spectral_rolloff_std'] = spectral_rolloffs.apply(lambda x: np.std(x))

    data = data.copy()
    return data

def chroma(df, sr=22050, n_fft=2048, hop_length=512):
    data = df.copy()

    chroma_stft = data['wave'].apply(
        lambda y: librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    )
    
    # Compute mean and standard deviation of chroma features
    data['chroma_mean'] = chroma_stft.apply(lambda chroma: np.mean(chroma))
    data['chroma_std'] = chroma_stft.apply(lambda chroma: np.std(chroma))

    data = data.copy()
    return data

def pitch_features(df, sr=22050):
    data = df.copy()

    features = data['wave'].apply(lambda wave: extract_pitch_features(wave, sr=sr))

    # Convert the list of Series to a DataFrame and concatenate with the original data
    features_df = pd.DataFrame(features.tolist(), columns=['my_mean_pitch', 'median_pitch', 'pitch_std', 'voiced_frame_ratio', 'mean_voiced_prob'])
    
    data = pd.concat([data, features_df], axis=1)

    data = data.copy() #De-fragment the dataframe
    return data

# *** EXTRACTORS ***

def extract_silence_durations(y, sr, num_frames=20, top_db=20):

    # Convert audio amplitudes to a numpy array
    if isinstance(y, list):
        y = np.array(y)

    # Calculate the total number of samples
    total_samples = len(y)

    # Calculate the frame duration in seconds
    frame_duration = total_samples / (num_frames * sr)
    frame_size = int(frame_duration * sr)  # Number of samples per frame
    
    silence_durations = np.zeros(num_frames)

    for i in range(num_frames):
        # Get the current frame
        start = i * frame_size
        end = start + frame_size
        frame = y[start:end]

        # Use librosa to detect non-silent intervals in the current frame
        non_silent_intervals = librosa.effects.split(frame, top_db=top_db)

        # Calculate the total silence duration in the current frame
        silence_duration = 0.0

        if len(non_silent_intervals) == 0:
            # Entire frame is silent
            silence_duration = frame_duration * sr
        else:
            # Calculate silence duration based on non-silent intervals
            for j in range(len(non_silent_intervals) - 1):
                silence_duration += (non_silent_intervals[j + 1][0] - non_silent_intervals[j][1]) / sr
            
            # Check if the frame ends with silence
            if non_silent_intervals[-1][1] < len(frame):
                silence_duration += (len(frame) - non_silent_intervals[-1][1]) / sr

        silence_durations[i] = silence_duration

    return silence_durations


def extract_pitch_features(wave, sr=22050, fmin=75, fmax=800, frame_length=2048, hop_length=512):

    # TODO Optimize with librosa.piptrack

    f0, voiced_flag, voiced_prob = librosa.pyin(y=wave, sr=sr, fmin=fmin, fmax=fmax, frame_length=frame_length, hop_length=hop_length, fill_na=None)

    mean_pitch = np.mean(f0[voiced_flag]) if np.any(voiced_flag) else 0
    median_pitch = np.median(f0[voiced_flag]) if np.any(voiced_flag) else 0
    pitch_std = np.std(f0[voiced_flag]) if np.any(voiced_flag) else 0
    voiced_frame_ratio = np.sum(voiced_flag) / len(voiced_flag)
    mean_voiced_prob = np.mean(voiced_prob[voiced_flag]) if np.any(voiced_flag) else 0

    features = np.array([mean_pitch, median_pitch, pitch_std, voiced_frame_ratio, mean_voiced_prob])
    
    return features