"""
Performance Estimation Strategy
trains the architectures created in search_space
"""
from typing import List
import keras


class Trainer(object):

    def __init__(self):
        self.dataset_path = "/home/amirhossein/Codes/Project/Dataset/Dataset_678/dataset_openclose_678"
        self.model_validation_split = 0.1
        self.model_batch_size = 10
        self.model_epochs = 1
        self.verbose = 0
        self.train_batch = None
        self.validation_batch = None
        self.read_dataset()

    def read_dataset(self):
        data_generator = keras.preprocessing.image.ImageDataGenerator(
            validation_split=self.model_validation_split,
            preprocessing_function=keras.applications.mobilenet.preprocess_input)
        self.train_batch = data_generator.flow_from_directory(self.dataset_path, target_size=(128, 128),
                                                              batch_size=self.model_batch_size, subset='training')
        self.validation_batch = data_generator.flow_from_directory(self.dataset_path, target_size=(128, 128),
                                                                   batch_size=self.model_batch_size,
                                                                   subset='validation',
                                                                   shuffle=False)

    def train_models(self, samples: List[List[int]], architectures: List[keras.models.Sequential]):
        epoch_performance = []
        for i, model in enumerate(architectures):
            # print(samples[i])
            # print(model.summary())
            history = model.fit(self.train_batch, steps_per_epoch=len(self.train_batch)/4,
                                validation_data=self.validation_batch, validation_steps=len(self.validation_batch),
                                epochs=self.model_epochs, verbose=self.verbose)

            epoch_performance.append([samples[i], history.history['val_accuracy'][0]])

        return epoch_performance
