"""
Performance Estimation Strategy
trains the architectures created in search_space
"""
import config
from latency_predictor import LatencyPredictor
import time
import numpy as np
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras


class Trainer(object):

    def __init__(self, tokens):
        self.dataset_path = config.trainer["dataset_path"]
        self.model_validation_split = config.trainer["model_validation_split"]
        self.model_batch_size = config.trainer["model_batch_size"]
        self.model_epochs = config.trainer["model_epochs"]
        self.verbose = config.trainer["verbose"]
        self.image_size = config.emnas["model_input_shape"][:2]
        self.train_batch = None
        self.validation_batch = None
        self.read_dataset()
        self.hardware = config.trainer["hardware"]  # sipeed, jevois
        self.outlier_limit = config.latency_predictor["outlier_limit"]
        self.acc_model = keras.models.load_model(config.trainer["predictor_path"])
        self.tokens = tokens
        self.len_search_space = len(tokens) + 1
        self.end_token = list(tokens.keys())[-1]

    def read_dataset(self):
        data_generator = keras.preprocessing.image.ImageDataGenerator(
            validation_split=self.model_validation_split,
            preprocessing_function=keras.applications.mobilenet.preprocess_input)
        self.train_batch = data_generator.flow_from_directory(self.dataset_path, target_size=self.image_size,
                                                              batch_size=self.model_batch_size, subset='training')
        self.validation_batch = data_generator.flow_from_directory(self.dataset_path, target_size=self.image_size,
                                                                   batch_size=self.model_batch_size,
                                                                   subset='validation',
                                                                   shuffle=False)

    def train_models(self, architectures):
        accuracies = []
        for model in architectures:
            if model is None:
                accuracy = 0.4  # assign bad reward
            else:
                history = model.fit(self.train_batch, steps_per_epoch=len(self.train_batch),
                                    validation_data=self.validation_batch, validation_steps=len(self.validation_batch),
                                    epochs=self.model_epochs, verbose=self.verbose)
                accuracy = round(history.history['val_accuracy'][-1], 3)
                accuracies.append(accuracy)
        return accuracies

    def performance_estimate(self, sequence):
        if self.end_token in sequence:
            sequence = np.append(sequence, 0)
        else:
            sequence = np.append(sequence, self.end_token)  # adding end token

        seq_hot = keras.utils.to_categorical(sequence, num_classes=self.len_search_space)[np.newaxis]
        acc = self.acc_model.predict(seq_hot)[0][0]
        return round(acc, 3)
