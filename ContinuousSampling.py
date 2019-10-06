import librosa.display
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np
import time
# import scipy.io.wavfile
# import wavio
#
# start = time.time()
#
# sampling_rate = 22050;
#
# # load audio file
# f = "bp1-03-002.wav"
#
# # data, sampling_rate = librosa.load("./all_samples/test/" + f)
# # rate, data = scipy.io.wavfile.read("./all_samples/test/" + f)
# data = wavio.read("./all_samples/test/" + f)
# data = np.delete(data, 1, 0)
# end = time.time()
# print(end - start)
#
# # create figure, remove borders
# fig = plt.figure(figsize=(12, 4))
# ax = fig.add_axes([0, 0, 1, 1])
# ax.axis('off')
# # plt.title(f)
#
#
# S = librosa.feature.melspectrogram(y=np.ndarray.astype(data.data, float), sr=sampling_rate)
# librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
# spec_file_name = "./specs/test/" + f[:-4] + ".png"
#
#
# spectral_centroid = np.average(librosa.feature.spectral_centroid(data, sampling_rate))
# zero_crossing_rate = np.average(librosa.feature.zero_crossing_rate(data, sampling_rate))
# rms = np.average(librosa.feature.rms(y=data))
#
#
# plt.savefig(spec_file_name)
# img = Image.open(spec_file_name)
# resolution = (240, 160)
# img = img.resize(resolution)
# img.save(spec_file_name)
# plt.close()
#


import pyaudio
import numpy as np

RATE = 22050
CHUNK = 240

p = pyaudio.PyAudio()

player = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, output=True,
                frames_per_buffer=CHUNK)
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

for i in range(int(20*RATE/CHUNK)): #do this for 10 seconds
    player.write(np.fromstring(stream.read(CHUNK), dtype=np.int16))
    # data = np.fromstring(stream.read(CHUNK), dtype=np.int16)
    # print(data)
    # time.sleep(5)
    # S = librosa.feature.melspectrogram(y=np.fromstring(stream.read(CHUNK)), sr=RATE)

stream.stop_stream()
stream.close()
p.terminate()
