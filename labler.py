from keras.models import Model, model_from_json
from keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

successes, fails = 0, 0
# make a model and load weights

model = model_from_json(open('model.json').read())
model.load_weights('weights-improvement-99-0.91.hdf5')

# load in a test image and it's parameters
metaIn = pd.read_csv("specs/test_metadata.csv", index_col="Name")
imgs = []
metas = []
outs = []
count = 0;
for root, dirs, files in os.walk("./specs/test"):
    for file in files:
        count += 1;
        img_name = file
        img = load_img("specs/test/" + img_name)
        img_array = img_to_array(img)
        # need to either load in or calculate the other usful data for waveform

        imgs.append(img_array)

        row = metaIn.loc[img_name, :]

        metas.append(row.iloc[1:4].values.reshape(3,))
        outs.append(row.iloc[4:].values.reshape(3,))

        # evaluate it with the model

        pred = model.predict(
            [np.expand_dims(img_array, axis=0), row.iloc[1:4].values.reshape(1, 3)])

        idx = np.argmax(pred, axis=1)
        # check nothing
        if(row.Nothing == 1 and idx == 0):
            print('Not digging: Success!')
            successes += 1
            # check BP1
        elif (row.BP1 == 1 and idx == 1):
            print('Digging soft soil: Success!')
            successes += 1
        elif (row.BP2 == 1 and idx == 2):
            print('Digging gravel: Success!')
            successes += 1
        else:
            print("FAILURE!")
            fails += 1

        # print(file)
        # print(pred)
print("count" + str(count))
print("count" + str(len(imgs)))
print("Performance: " + str(successes/(successes + fails)*100))


pred = model.predict([imgs, metas])
pred = np.argmax(pred, axis=1)
outs = np.argmax(outs, axis=1)

cm = confusion_matrix(outs, pred)

print(classification_report(outs, pred))

df_cm = pd.DataFrame(
    cm
)
fig = plt.figure(figsize=(3,3))
try:
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
except ValueError:
    raise ValueError("Confusion matrix values must be integers.")
heatmap.yaxis.set_ticklabels(
    heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=12)
heatmap.xaxis.set_ticklabels(
    heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=12)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
