from keras.models import Model, model_from_json
from keras.preprocessing.image import img_to_array, load_img
import pandas as pd
import numpy as np
import os

#make a model and load weights

model = model_from_json(open('model.json').read())
model.load_weights('weights-improvement-99-0.91.hdf5')

#load in a test image and it's parameters
metaIn = pd.read_csv("specs/test_metadata.csv", index_col="Name")
for root, dirs, files in os.walk("./specs/test"):
    for file in files:
        img_name = file
        img = load_img("specs/test/" + img_name)
        img_array = img_to_array(img)
        #need to either load in or calculate the other usful data for waveform

        row = metaIn.loc[img_name, : ]

        # evaluate it with the model

        pred = model.predict([np.expand_dims(img_array, axis=0), row.iloc[1:4].values.reshape(1, 3)])

        print(file)
        print(pred)



