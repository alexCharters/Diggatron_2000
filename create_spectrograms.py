import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image

class create_data():
    """create_data.py: Converts all of the .wav files into spectrograms"""

    @staticmethod
    def create_spectrograms():
        """Creates spectrograms from all of the .wav files in /all_sounds"""
        for root, dirs, files in os.walk("./all_samples/"):
            for f in files:
                # load audio file
                data, sampling_rate = librosa.load("./all_samples/"+f)
                # create figure, remove borders
                fig = plt.figure(figsize=(12, 4))
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

    @staticmethod
    def get_spectral_centroids():
        for root, dirs, files in os.walk("./all_samples/"):
            for f in files:
                # load audio file
                data, sampling_rate = librosa.load("./all_samples/"+f)

                S = librosa.feature.melspectrogram(y=data, sr=sampling_rate)
                spectral_centroid = librosa.feature.spectral_centroid(S, sampling_rate)


                plt.savefig(spec_file_name)
                img = Image.open(spec_file_name)
                resolution = (240, 160)
                img = img.resize(resolution)
                img.save(spec_file_name)
                plt.close()

    @staticmethod
    def generate_metadata():
        cols = ['Name', 'Spectral_Center', 'Cross_Rate', 'Nothing', 'BP1', 'BP2']
        metadata = pd.DataFrame(columns=cols)
        for root, dirs, files in os.walk("./all_samples/"):
            for f in files:
                data, sampling_rate = librosa.load("./all_samples/"+f)
                spectral_centroid = np.average(librosa.feature.spectral_centroid(data, sampling_rate))
                zero_crossing_rate = np.average(librosa.feature.zero_crossing_rate(data, sampling_rate))

                label = f[0:3]
                if label == "bp1":
                    label = [0, 1, 0]
                elif label == "bp2":
                    label = [0, 0, 1]
                else:
                    label = [1, 0, 0]

                row = pd.DataFrame([f[:-4] + ".png", spectral_centroid, zero_crossing_rate, label[0], label[1], label[2]])
                row = row.T
                row.columns = cols
                metadata = metadata.append(row)
        print(metadata)
        metadata.to_csv('./specs/metadata.csv')


if __name__ == "__main__":
    create_data.generate_metadata()
