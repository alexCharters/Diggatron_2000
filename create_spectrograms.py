import librosa.display
import matplotlib.pyplot as plt
import numpy as np
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
        pass


if __name__ == "__main__":
    create_data.generate_metadata()