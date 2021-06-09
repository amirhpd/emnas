"""
Performance Estimation Strategy
trains the architectures created in search_space
"""
import config
import time
import numpy as np
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras


class Trainer(object):

    def __init__(self, tokens):
        self.dataset_path_fast = config.trainer["dataset_path_fast"]
        self.dataset_path_full = config.trainer["dataset_path_full"]
        self.model_validation_split = config.trainer["model_validation_split"]
        self.model_batch_size = config.trainer["model_batch_size"]
        self.model_epochs_fast = config.trainer["model_epochs_fast"]
        self.model_epochs_full = config.trainer["model_epochs_full"]
        self.verbose = config.trainer["verbose"]
        self.image_size = config.emnas["model_input_shape"][:2]
        self.multi_obj_weight = config.trainer["multi_obj_weight"]
        self.train_batch = None
        self.validation_batch = None
        self.acc_model = keras.models.load_model("predictors/accuracy_predictor.h5")
        self.lat_model = keras.models.load_model("predictors/latency_predictor.h5")
        self.tokens = tokens
        self.len_search_space = len(tokens) + 1
        self.end_token = list(tokens.keys())[-1]

    def read_dataset(self, train_mode):
        dataset_path = self.dataset_path_full if train_mode == "full" else self.dataset_path_fast
        data_generator = keras.preprocessing.image.ImageDataGenerator(
            validation_split=self.model_validation_split,
            preprocessing_function=keras.applications.mobilenet.preprocess_input)
        self.train_batch = data_generator.flow_from_directory(dataset_path, target_size=self.image_size,
                                                              batch_size=self.model_batch_size, subset='training')
        self.validation_batch = data_generator.flow_from_directory(dataset_path, target_size=self.image_size,
                                                                   batch_size=self.model_batch_size,
                                                                   subset='validation',
                                                                   shuffle=False)

    def train_models(self, architectures, train_mode):
        self.read_dataset(train_mode)
        epochs = self.model_epochs_full if train_mode == "full" else self.model_epochs_fast
        accuracies = []
        for model in architectures:
            if model is None:
                accuracy = 0.4  # assign bad reward
            else:
                history = model.fit(self.train_batch, steps_per_epoch=len(self.train_batch),
                                    validation_data=self.validation_batch, validation_steps=len(self.validation_batch),
                                    epochs=epochs, verbose=self.verbose)
                accuracy = history.history['val_accuracy'][-1]
                accuracies.append(accuracy)
        return accuracies

    def performance_estimate(self, sequence):
        if self.end_token in sequence:
            sequence = np.append(sequence, 0)
        else:
            sequence = np.append(sequence, self.end_token)  # adding end token

        seq_hot = keras.utils.to_categorical(sequence, num_classes=self.len_search_space)[np.newaxis]
        try:
            acc = self.acc_model.predict(seq_hot)[0][0]
            lat = self.lat_model.predict(seq_hot)[0][0]
        except ValueError as e:
            raise ValueError("Predictor models does not match the generated sequence.", e)
        return round(acc, 3), round(lat, 3)

    def multi_objective_reward(self, accuracy, latency):
        acc = np.clip(accuracy, 0.4, 1)
        latency = np.clip(latency, 0, 1000)
        lat = (-6e-4 * latency) + 1
        reward = np.average([acc, lat], weights=[self.multi_obj_weight, 1])
        return round(reward, 4)
