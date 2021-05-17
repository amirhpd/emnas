"""
Search Space
creates different possibilities of NN as a dict of tokens and layer info
translates a sequence of tokens to layer info
converts a token sequence to keras model
"""
import itertools
from typing import List, Dict, Tuple
import config
import json
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
from keras import optimizers
from keras.models import Sequential
import tensorflow as tf
import collections


class SearchSpace(object):

    def __init__(self, model_output_shape):

        self.model_output_shape = model_output_shape
        self.model_dropout = config.search_space["model_dropout"]
        self.model_loss_function = config.search_space["model_loss_function"]
        self.model_optimizer = config.search_space["model_optimizer"]
        self.model_lr = config.search_space["model_lr"]
        self.model_decay = config.search_space["model_decay"]
        self.model_metrics = config.search_space["model_metrics"]
        self.search_mode = config.emnas["search_mode"]

    def generate_token(self) -> Dict:
        ranges = json.load(open("search_space.json"))
        nodes = ranges["nodes"]
        layers = ranges["layers"]
        filters = ranges["filters"]
        paddings = ranges["paddings"]
        activations = ranges["activations"]
        kernel_sizes = [tuple(i) for i in ranges["kernel_sizes"]]
        strides = [tuple(i) for i in ranges["strides"]]

        # keys as cont. int
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

        tokens = {**cnn_token, **dense_token}

        # keys as coded int
        # space_cnn = [layers, filters, kernel_sizes, strides, paddings, activations]
        # cnn_params = list(itertools.product(*space_cnn))
        # elements = [item for sublist in space_cnn for item in sublist]
        # elements = list(collections.OrderedDict.fromkeys(elements))
        # elements_dict = dict(zip(elements, range(10, len(elements)+11)))
        # cnn_token = {}
        # for layer in cnn_params:
        #     token = ""
        #     for param in layer:
        #         letter = str(elements_dict[param])
        #         token += letter
        #     cnn_token.update({int(token): layer})
        #
        # space_dense = [["Dense"], nodes, activations]
        # dense_params = list(itertools.product(*space_dense))
        # elements_d = [item for sublist in space_dense for item in sublist]
        # elements_d = list(collections.OrderedDict.fromkeys(elements_d))
        # elements_d_dict = dict(zip(elements_d, range(10, len(elements_d)+11)))
        # dense_token = {}
        # for layer in dense_params:
        #     token = ""
        #     for param in layer:
        #         letter = str(elements_d_dict[param])
        #         token += letter
        #     dense_token.update({int(token): layer})
        #
        # tokens = {**cnn_token, **dense_token, 1: ("dropout", self.model_dropout)}
        # if self.model_output_shape == 1:
        #     tokens[2] = (1, "sigmoid")
        # else:
        #     tokens[2] = (self.model_output_shape, "softmax")

        return tokens

    def translate_sequence(self, sequence):
        token = self.generate_token()
        translated = []
        for layer in sequence:
            layer_info = token[layer]
            translated.append(layer_info)
        return translated

    def create_model(self, sequence, model_input_shape):
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
        model.compile(loss=self.model_loss_function, optimizer=optimizer, metrics=self.model_metrics)

        return model

    def create_models(self, samples: List, model_input_shape: Tuple) -> List:
        architectures = []
        for sequence in samples:
            try:
                architecture = self.create_model(sequence=sequence, model_input_shape=model_input_shape)
                architectures.append(architecture)
            except ValueError as e:
                if self.search_mode == "ff":
                    print("Skipped:", sequence, "due to:", e)
                architectures.append(None)

        return architectures
