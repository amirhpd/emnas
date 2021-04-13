"""
Performance Estimation Strategy
trains the architectures created in search_space
"""
import config
from latency_predictor import LatencyPredictor
import time
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras


class Trainer(object):

    def __init__(self):
        self.dataset_path = config.trainer["dataset_path"]
        self.model_validation_split = config.trainer["model_validation_split"]
        self.model_batch_size = config.trainer["model_batch_size"]
        self.model_epochs = config.trainer["model_epochs"]
        self.verbose = config.trainer["verbose"]
        self.image_size = config.emnas["model_input_shape"][:2]
        self.train_batch = None
        self.validation_batch = None
        self.read_dataset()
        self.latency_predictor = LatencyPredictor()
        self.hardware = config.trainer["hardware"]  # sipeed, jevois

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

    def train_models(self, samples, architectures):
        epoch_performance = {}
        for i, model in enumerate(architectures):
            t1 = time.time()
            history = model.fit(self.train_batch, steps_per_epoch=len(self.train_batch),
                                validation_data=self.validation_batch, validation_steps=len(self.validation_batch),
                                epochs=self.model_epochs, verbose=self.verbose)

            t2 = round(time.time() - t1, 2)
            acc = round(history.history['val_accuracy'][-1], 2)
            latency = round(self.latency_predictor.inference(sequence=samples[i], hardware=self.hardware), 2)
            print("Sequence:", samples[i], "Accuracy:", acc, "Latency:", latency, "ms")
            # epoch_performance.append([samples[i], acc])
            epoch_performance[tuple(samples[i])] = (acc, latency)

        return epoch_performance
