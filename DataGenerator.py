import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.DateFrame import to_numpy

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
        return math.floor(len([name for name in os.listdir('all_samples') if os.path.isfile('all_samples'+'//'+name)])/self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        img_files_temp = [self.img_files[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(img_files_temp)

        return X, y

    def __data_generation(img_file):
        #returns the properties of sound bit if the following form
        # (img_array, [specter center, row]), (expected output of neural net for wavform) 
        img_array = img_to_array((self.imgs_dir+'//'+img_file))

        metaData = self.metadata_dataframe.loc[img_file]['Center','Rate'].to_numpy()
        output_data = self.output_dataframe.loc[img_file]['Nothing', 'BP1', 'BP2'].to_numpy()
        
        
        return (img_array,metaData), output_data