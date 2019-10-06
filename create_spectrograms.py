import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
"""create_data.py: Converts all of the .wav files into spectrograms and finds other features for each clip"""

# n = 1
for root, dirs, files in os.walk("./all_samples/"):
    for f in files:
        # load audio file
        data, sampling_rate = librosa.load("./all_samples/"+f)
        # create figure, remove borders
        fig = plt.figure(figsize=(12, 4))
        # n = n+1
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        # plt.title(f)
        S = librosa.feature.melspectrogram(y=data, sr=sampling_rate)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
        spec_file_name = "./specs/" + f[:-4] + ".png"

        plt.savefig(spec_file_name)
        img = Image.open(spec_file_name)
        resolution = (240, 160)
        img = img.resize(resolution)
        img.save(spec_file_name)
        plt.close()
