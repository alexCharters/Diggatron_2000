from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img
import pandas as pd
import numpy as np


#load in the weights from a trained set
model = create_model()
model.load_weights("weights-improvement-best.hdf5")

#load in a test image and it's parameters
img_name = 'bp1-01-001.png'
img = load_img("specs/test/" + img_name)
img_array = img_to_array(img)
#need to either load in or calculate the other usful data for waveform


#evaluate it with the model

pred = model.predict(img)



