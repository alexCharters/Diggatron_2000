import librosa.feature
import numpy as np
import os
import pandas as pd

cols = ['Name', 'Spectral_Center', 'Cross_Rate', 'Spectral_Rolloff', 'Spectral_Flatness', 'Spectral_Bandwidth', 'rms']

# TRAINING
metadata = pd.DataFrame(columns=cols)
for root, dirs, files in os.walk("./all_samples/train"):
    for f in files:
        data, sampling_rate = librosa.load("./all_samples/train/" + f)
        print(sampling_rate)
        spectral_centroid = np.average(librosa.feature.spectral_centroid(data, sampling_rate))
        zero_crossing_rate = np.average(librosa.feature.zero_crossing_rate(data, sampling_rate))
        spectral_rolloff = np.average(librosa.feature.spectral_rolloff(data, sampling_rate))
        spectral_flatness = np.average(librosa.feature.spectral_flatness(y=data))
        spectral_bandwidth = np.average(librosa.feature.spectral_bandwidth(data, sampling_rate))
        S = librosa.feature.melspectrogram(y=data, sr=sampling_rate)
        rms = np.average(librosa.feature.rms(y=data))


        label = f[0:3]
        if label == "bp1":
            label = [0, 1, 0]
        elif label == "bp2":
            label = [0, 0, 1]
        else:
            label = [1, 0, 0]

        row = pd.DataFrame([f[:-4] + ".png", spectral_centroid, zero_crossing_rate, spectral_rolloff, spectral_flatness, spectral_bandwidth, rms])
        row = row.T
        row.columns = cols
        metadata = metadata.append(row)
print(metadata)
metadata.to_csv('./sounds/experimental_metadata.csv')
