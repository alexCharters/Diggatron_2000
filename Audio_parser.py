import sounddevice as sd

rate = 48000
time = 0.5

sample = sd.rec((rate*time), samplerate=rate, channels=2, )