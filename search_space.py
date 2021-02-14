"""
Search Space
creates different possibilities of NN as a dict of tokens and layer info
translates a sequence of tokens to layer info
converts a token sequence to keras model
"""
from keras import optimizers
from keras.models import Sequential
import keras
import itertools
from typing import List, Dict, Tuple


class SearchSpace(object):

    def __init__(self, model_output_shape):

        self.model_output_shape = model_output_shape
        self.model_dropout = 0.2
        self.model_loss_function = "categorical_crossentropy"
        self.model_optimizer = "Adam"
        self.model_lr = 0.001
        self.model_decay = 0.0
        self.metrics = ["accuracy"]

    def generate_token(self) -> Dict:
        nodes = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
        layers = ["Conv2D", "DepthwiseConv2D"]
        filters = [8, 16, 24, 32, 40]
        kernel_sizes = [(1, 1), (2, 2), (3, 3)]
        strides = [(1, 1), (2, 2), (3, 3)]
        paddings = ["valid", "same"]
        activations = ["sigmoid", "tanh", "relu"]

        # keys as int
        cnn_params = list(itertools.product(*[layers, filters, kernel_sizes, strides, paddings, activations]))
        cnn_count = range(1, len(cnn_params)+1)
        cnn_token = dict(zip(cnn_count, cnn_params))

        dense_params = list(itertools.product(*[["Dense"], nodes, activations]))
        dense_count = range(cnn_count[-1]+1, cnn_count[-1] + len(dense_params)+1)
        dense_token = dict(zip(dense_count, dense_params))

        dense_token[dense_count[-1]+1] = ("dropout", self.model_dropout)
        if self.model_output_shape == 1:
            dense_token[dense_count[-1]+2] = (1, "sigmoid")
        else:
            dense_token[dense_count[-1]+2] = (self.model_output_shape, "softmax")

        return {**cnn_token, **dense_token}

    def translate_sequence(self, sequence):
        token = self.generate_token()
        translated = []
        for layer in sequence:
            layer_info = token[layer]
            translated.append(layer_info)
        return translated

    def create_model(self, sequence: List[int], model_input_shape: Tuple) -> Sequential:
        layers_info = self.translate_sequence(sequence)
        model = Sequential()
        if layers_info[0][0] == "Conv2D":
            model.add(keras.layers.Conv2D(filters=layers_info[0][1], kernel_size=layers_info[0][2],
                                          strides=layers_info[0][3], padding=layers_info[0][4],
                                          activation=layers_info[0][5], input_shape=model_input_shape))
        elif layers_info[0][0] == "DepthwiseConv2D":
            model.add(keras.layers.DepthwiseConv2D(kernel_size=layers_info[0][2], strides=layers_info[0][3],
                                                   padding=layers_info[0][4], activation=layers_info[0][5],
                                                   input_shape=model_input_shape))
        else:
            raise ValueError(f"emnas: Layer Type Unknown {str(layers_info[0])}")

        flatten_flag = False
        for layer in layers_info[1:-1]:
            if layer[0] == "Conv2D":
                model.add(keras.layers.Conv2D(filters=layer[1], kernel_size=layer[2], strides=layer[3],
                                              padding=layer[4], activation=layer[5]))
            elif layer[0] == "DepthwiseConv2D":
                model.add(keras.layers.DepthwiseConv2D(kernel_size=layer[2], strides=layer[3], padding=layer[4],
                                                       activation=layer[5]))
            elif layer[0] == "Dense":
                if not flatten_flag:
                    model.add(keras.layers.Flatten())
                    flatten_flag = True

                model.add(keras.layers.Dense(units=layer[1], activation=layer[2]))
            elif layer[0] == "dropout":
                model.add(keras.layers.Dropout(rate=layer[1]))
            else:
                raise ValueError(f"emnas: Layer Type Unknown {str(layer)}")

        if not flatten_flag:
            model.add(keras.layers.Flatten())
            flatten_flag = True
        model.add(keras.layers.Dense(units=layers_info[-1][0], activation=layers_info[-1][1]))

        if self.model_output_shape == 1:
            self.model_loss_function = "binary_crossentropy"
        optimizer = getattr(optimizers, self.model_optimizer)(lr=self.model_lr, decay=self.model_decay)
        model.compile(loss=self.model_loss_function, optimizer=optimizer, metrics=self.metrics)

        return model

    def create_models(self, samples: List, model_input_shape: Tuple) -> List:
        architectures = []
        for sequence in samples:
            architecture = self.create_model(sequence=sequence, model_input_shape=model_input_shape)
            architectures.append(architecture)
        return architectures
