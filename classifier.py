from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Concatenate, Input, Dense
from keras.models import Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers.core import Activation, Flatten, Dropout, Dense

datagen = ImageDataGenerator(
    rotation_range=0,
    width_shift_range = 0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    fill_mode='nearest'
)

img = load_img('')

inputs = Input(shape=(W,L,3))
Conv2D1 = Conv2D(32, (3,3), padding="same")(inputs)
Activation1 = Activation("relu")(Conv2D1)
BatchNormal1 = BatchNormalization(axiis = -1)(Activation1)
MaxPool1 = MaxPooling2D(pool_size=(3, 3))(BatchNormal1)


output = Dense(3, activation="softmax")
