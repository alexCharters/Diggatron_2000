import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import seaborn as sns
import glob
import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, Model
from keras.layers import Concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from progress.bar import Bar
image_list = []
name_list = []
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--learnrate", type=float, default=0.001,
                help="initial learning rate")
ap.add_argument("-e", "--epochs", type=int, default=50,
                help="number of epochs")
ap.add_argument("-b", "--batchsize", type=int, default=32,
                help="path to input dataset (i.e., directory of images)")
ap.add_argument("-spe", "--steps_per_epoch", type=int, default=500,
                help="how many batch steps occur per epoch")
args = vars(ap.parse_args())
W = 150
L = int(W*0.75)
print("[info] loading data")
path, dirs, files = next(os.walk("import/HAM/"))
file_num = len(files)-1
bar = Bar('Images Loaded', max=file_num)
for filename in glob.glob('import/HAM/*.jpg'):
    temp = Image.open(filename)
    keep = temp.copy()
    keep = keep.resize((W, L), Image.LANCZOS)
    keep_arr = np.array(keep)
    flattened_arr = keep_arr.reshape(W*L, 3)
    flattened_arr = flattened_arr/255
    flattened_arr = flattened_arr.astype(np.float32)
    name_list.append(filename[11:-4])
    image_list.append(flattened_arr)
    bar.next()
    temp.close()
df = pd.DataFrame.from_records(image_list)
name_df = pd.Series((x for x in name_list))
df['image_id'] = name_df
meta = pd.read_csv('import/HAM/HAM10000_metadata.csv')

merged_df = pd.merge(df, meta, on="image_id")
merged_df = pd.concat([merged_df, pd.get_dummies(
    merged_df['localization'], prefix='localization')], axis=1)
merged_df = pd.concat([merged_df, pd.get_dummies(
    merged_df['sex'], prefix='sex')], axis=1)
merged_df.drop(['localization'], axis=1, inplace=True)
merged_df.drop(['sex'], axis=1, inplace=True)

print("[info] normalizing data")
all_data = merged_df
train, test = train_test_split(all_data, test_size=0.2)

y_train = train['dx']
y_test = test['dx']
del train['dx']
del test['dx']
X_train = train
X_test = test

del X_train['image_id']
del X_train['lesion_id']
del X_train['dx_type']
#del X_train['localization']
del X_test['image_id']
del X_test['lesion_id']
del X_test['dx_type']
#del X_test['localization']

X_train['age'] = X_train['age']/100
X_test['age'] = X_test['age']/100

del df
del merged_df
del all_data
del train
del test

print("[info] creating image model")
model = Sequential()

model.add(Conv2D(32, (3, 3), padding="same", input_shape=(W, L, 3)))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(rate=0.25))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

print("[info] creating metadata model")
metadata_model = Sequential()
metadata_model.add(Dense(32, input_dim=19, activation="relu"))
metadata_model.add(Dense(64))

print("[info] merging models")
x = Concatenate()([model.output, metadata_model.output])
x = Dense(7)(x)
out = Activation("softmax")(x)

merged_model = Model([model.input, metadata_model.input], out)

print(merged_model.summary())

print("[info] compiling")
EPOCHS = args["epochs"]
INIT_LR = args["learnrate"]
BS = args["batchsize"]
IMAGE_DIMS = (W, L, 3)
STEPS_PER_EPOCH = args["steps_per_epoch"]

lb = LabelBinarizer()
labels = lb.fit_transform(np.array(y_train))
labels_test = lb.fit_transform(np.array(y_test))

opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
merged_model.compile(loss="categorical_crossentropy",
                     optimizer=opt, metrics=["accuracy"])

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1,
                         shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

print("[info] prepping inputs")
trainInput = np.concatenate(
    np.array(X_train.iloc[:, 0:L*W]).ravel()).reshape(X_train.shape[0], W, L, 3)
testInput = np.concatenate(
    np.array(X_test.iloc[:, 0:L*W]).ravel()).reshape(X_test.shape[0], W, L, 3)

print("[info] training model")
train = merged_model.fit_generator(
    aug.flow([trainInput, X_train.iloc[:, L*W:]], labels, batch_size=BS),
    validation_data=([testInput, X_test.iloc[:, L*W:]], labels_test),
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    verbose=1)

print("[info] evaluating and saving model")
test_imgs = np.concatenate(
    np.array(X_test.iloc[:, 0:L*W]).ravel()).reshape(X_test.shape[0], W, L, 3)
scores = merged_model.evaluate(test_imgs, labels_test, verbose=1)

merged_model.save("whole_model.h5")

f = open("binarizer.pkl", "wb")
f.write(pickle.dumps(lb))
f.close()

print("Saved model to disk")

predicted_classes = merged_model.predict_classes(test_imgs)

cm = confusion_matrix([np.where(r == 1)[0][0]
                       for r in labels_test], predicted_classes)
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True)
sns.show()
sns.savefig("confusion_matrix.png")

plt.style.use("ggplot")
plt.figure(figsize=(14, 10))
N = EPOCHS
plt.plot(np.arange(0, N), train.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), train.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), train.history["acc"], label="acc")
plt.plot(np.arange(0, N), train.history["val_acc"], label="acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.show()
plt.savefig('accuracy_and_loss.png')
