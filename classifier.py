from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Input, Dense
from keras.layers import concatenate
from keras.models import Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

from datetime import datetime

from DataGenerator import DataGenerator

W = 240
H = 160
INIT_LR = .01
EPOCHS = 80
BATCH_SIZE = 32

# train_datagen = ImageDataGenerator(
#     rotation_range=0,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     zoom_range=0.2,
#     fill_mode='nearest'
# )

# test_datagen = ImageDataGenerator(
#     rotation_range=0,
#     zoom_range=0.1,
#     fill_mode='nearest'
# )

#img = load_img('')
# conv layers for image
imgInput = Input(shape=(W, H, 3))

conv2D = Conv2D(96, (11, 11), strides=4, padding="same")(imgInput)
maxPool = MaxPooling2D(3, 2)(conv2D)

conv2D1 = Conv2D(256, (5, 5), padding="same")(maxPool)
activation1 = Activation("relu")(conv2D1)
batchNormal1 = BatchNormalization(axis=-1)(activation1)
maxPool1 = MaxPooling2D(3, 2)(batchNormal1)
dropOut1 = Dropout(rate=0.25)(maxPool1)

conv2D2 = Conv2D(384, (3, 3), padding="same")(dropOut1)
maxPool2 = MaxPooling2D(3, 2)(conv2D2)

conv2D3 = Conv2D(256, (3, 3), padding="same")(maxPool2)
activation3 = Activation("relu")(conv2D3)
batchNormal3 = BatchNormalization(axis=-1)(activation3)
maxPool3 = MaxPooling2D(3, 2)(batchNormal3)
dropOut3 = Dropout(rate=0.25)(maxPool3)

flat = Flatten()(dropOut3)


# dense layers for extra data
extraInput = Input(shape=(3,))
extraDense1 = Dense(10, activation="relu")(extraInput)
merge = concatenate([flat, extraDense1])

Dense1 = Dense(4096, activation="relu")(merge)
Dense2 = Dense(4096, activation="relu")(Dense1)

output = Dense(3, activation="softmax")(Dense2)

model = Model(inputs=[imgInput, extraInput], outputs=output)

print(model.summary())
#hello this is my code. I can code good see Do_the_thing("motivation");
#plot_model(model, to_file='model.png')

#load data into a panda data frame
train_meta = pd.read_csv("specs/train_metadata.csv",usecols=['Name','Spectral_Center','Cross_Rate'])
train_outData = pd.read_csv("specs/train_metadata.csv", usecols=['Nothing','BP1','BP2'])

test_meta = pd.read_csv("specs/test_metadata.csv",usecols=['Name','Spectral_Center','Cross_Rate'])
test_outData = pd.read_csv("specs/test_metadata.csv", usecols=['Nothing','BP1','BP2'])
# print(meta)
# print(outData)


training_datagen = DataGenerator('specs/train', train_meta, train_outData, batch_size=BATCH_SIZE)
testing_datagen = DataGenerator('specs/test', test_meta, test_outData, batch_size=BATCH_SIZE)


opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# train_generator = train_datagen.flow_from_directory(
#     'specs/train',
#     target_size=(240, 160),
#     batch_size=BATCH_SIZE
# )

# test_generator = train_datagen.flow_from_directory(
#     'specs/test',
#     target_size=(240, 160),
#     batch_size=BATCH_SIZE
# )


model.fit_generator(
    generator=training_datagen,
    epochs=EPOCHS,
    validation_data=testing_datagen,
    #callbacks=[checkpoint]
)

#save final weights
now = datetime.now()
model.save_weights('final_weights_'+now.strftime("%m_%d_%Y_%H_%M_%S")+'.h5')

# evaluating model

# run on testing data
Y_pred = model.predict_generator(validation_generator, num_of_test_samples // BATCH_SIZE+1)
y_pred = np.argmax(Y_pred, axis=1)

#print(confusion_matrix([, , ], y_pred))