from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Input, Dense
from keras.layers import concatenate
from keras.models import Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers.core import Activation, Flatten, Dropout, Dense
import pandas as pd

W = 120
H = 60

datagen = ImageDataGenerator(
    rotation_range=0,
    width_shift_range = 0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    fill_mode='nearest'
)

#img = load_img('')
#conv layers for image
imgInput = Input(shape=(W,H,3))
conv2D1 = Conv2D(32, (3,3), padding="same")(imgInput)
activation1 = Activation("relu")(conv2D1)
batchNormal1 = BatchNormalization(axis = -1)(activation1)
maxPool1 = MaxPooling2D(pool_size=(3, 3))(batchNormal1)
dropOut1 = Dropout(rate=0.25)(maxPool1)

conv2D2 = Conv2D(32, (3,3), padding="same")(maxPool1)
activation2 = Activation("relu")(conv2D2)
batchNormal2 = BatchNormalization(axis = -1)(activation2)
maxPool2 = MaxPooling2D(pool_size=(3, 3))(batchNormal2)
dropOut2 = Dropout(rate=0.25)(maxPool2)

flat = Flatten()(maxPool2)


#dense layers for extra data
extraInput = Input(shape=(3,))
extraDense1 = Dense(10, activation="relu")(extraInput)

merge = concatenate([flat, extraDense1])

Dense1 = Dense(1024, activation="relu")(merge)
Dense2 = Dense(1024, activation="relu")(Dense1)

output = Dense(3, activation="softmax")(Dense2)

model = Model(inputs=[imgInput, extraInput], outputs=output)

print(model.summary())
#hello this is my code. I can code good see Do_the_thing("motivation");
plot_model(model, to_file='model.png')

#load data into a panda data frame
meta = pd.read_csv("metaData.csv",usecols=['Center','Rate'])
outData = pd.read_csv("metaData.csv", usecols=['Nothing','BP1','BP2'])
print(meta)
print(outData)