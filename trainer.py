"""
Performance Estimation Strategy
trains the architectures created in search_space
"""
import keras
import config


class Trainer(object):

    def __init__(self):
        self.dataset_path = config.trainer["dataset_path"]
        self.model_validation_split = config.trainer["model_validation_split"]
        self.model_batch_size = config.trainer["model_batch_size"]
        self.model_epochs = config.trainer["model_epochs"]
        self.verbose = config.trainer["verbose"]
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

    def train_models(self, samples, architectures):
        epoch_performance = {}
        for i, model in enumerate(architectures):
            history = model.fit(self.train_batch, steps_per_epoch=len(self.train_batch)/4,
                                validation_data=self.validation_batch, validation_steps=len(self.validation_batch),
                                epochs=self.model_epochs, verbose=self.verbose)

            acc = history.history['val_accuracy'][0]
            print("Sequence:", samples[i], "Accuracy:", acc)
            # epoch_performance.append([samples[i], acc])
            epoch_performance[tuple(samples[i])] = acc

        return epoch_performance
