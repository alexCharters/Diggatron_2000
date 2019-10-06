import os
import keras
import math
import threading
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pandas as pd
import numpy as np


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras."""

    def __init__(self, imgs_dir, metadata_dataframe, output_dataframe, batch_size=32):
        """Initialization.
        
        Args:
            imgs_dir: directory to images.
        """
        self.imgs_dir = imgs_dir
        self.metadata_dataframe = metadata_dataframe
        self.output_dataframe = output_dataframe
        self.batch_size = batch_size

    def __len__(self):
        """Denotes the number of batches per epoch."""
        # print("len: " + str(math.floor(len([name for name in os.listdir(self.imgs_dir) if os.path.isfile(self.imgs_dir+'//'+name)])/self.batch_size)-1)
        return math.floor(len([name for name in os.listdir(self.imgs_dir) if
                               os.path.isfile(self.imgs_dir + '//' + name)]) / self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        rows = self.metadata_dataframe.iloc[index * self.batch_size:(index + 1) * self.batch_size]
        names = rows['Name']

        rng = range(index * self.batch_size, (index + 1) * self.batch_size)
        img_files_temp = [names[k] for k in rng]
        # create batch item list
        img_batch_list = []
        meta_batch_list = []
        y_batch_list = []
        for img_file in img_files_temp:
            # Generate data
            print("IMAGE FILE:(")
            print(img_file)
            img, meta, y = self.__data_generation(img_file)
            img_batch_list.append(img)
            meta_batch_list.append(meta)
            y_batch_list.append(y)

        # batch_inputs = (img_batch_list, meta_batch_list)
        # return batch_inputs #, y_batch_list
        return [np.array(img),np.array(meta_batch_list)], np.array(y_batch_list)

    def __data_generation(self, img_file):
        # returns the properties of sound bit if the following form
        # (img_array, [specter center, row]), (expected output of neural net for wavform)
        img = load_img(self.imgs_dir + '//' + img_file)
        img_array = img_to_array(img)

        metaData = np.array([
            self.metadata_dataframe.loc[self.metadata_dataframe['Name'] == img_file]['Spectral_Center'].values[0],
            self.metadata_dataframe.loc[self.metadata_dataframe['Name'] == img_file]['Cross_Rate'].values[0]
        ])
        output_data = np.array([
            self.output_dataframe.loc[self.metadata_dataframe['Name'] == img_file]['Nothing'].values[0],
            self.output_dataframe.loc[self.metadata_dataframe['Name'] == img_file]['BP1'].values[0],
            self.output_dataframe.loc[self.metadata_dataframe['Name'] == img_file]['BP2'].values[0]
        ])

        return img_array, metaData, output_data
