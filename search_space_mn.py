"""
Search Space MobileNets
creates models that are similar to MobileNets
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


class SearchSpaceMn(object):

    def __init__(self, model_output_shape):

        self.model_output_shape = model_output_shape
        self.model_dropout = config.search_space["model_dropout"]
        self.model_loss_function = config.search_space["model_loss_function"]
        self.model_optimizer = config.search_space["model_optimizer"]
        self.model_lr = config.search_space["model_lr"]
        self.model_decay = config.search_space["model_decay"]
        self.model_metrics = config.search_space["model_metrics"]

    def generate_token(self) -> Dict:
        ranges = json.load(open("search_space_mn.json"))

        filters = ranges["Conv2D"]["filters"]
        paddings = ranges["Conv2D"]["paddings"]
        kernel_sizes = [tuple(i) for i in ranges["Conv2D"]["kernel_sizes"]]
        strides = [tuple(i) for i in ranges["Conv2D"]["strides"]]
        c2d_params = list(itertools.product(*[["Conv2D"], filters, kernel_sizes, strides, paddings]))
        c2d_count = range(1, len(c2d_params)+1)
        c2d_token = dict(zip(c2d_count, c2d_params))

        paddings = ranges["DepthwiseConv2D"]["paddings"]
        kernel_sizes = [tuple(i) for i in ranges["DepthwiseConv2D"]["kernel_sizes"]]
        strides = [tuple(i) for i in ranges["DepthwiseConv2D"]["strides"]]
        dw_params = list(itertools.product(*[["DepthwiseConv2D"], kernel_sizes, strides, paddings]))
        dw_count = range(c2d_count[-1]+1, c2d_count[-1] + len(dw_params)+1)
        dw_token = dict(zip(dw_count, dw_params))

        dropouts = ranges["General"]["dropout"]
        zeropads = ranges["General"]["zeropadding"]
        general_params = []
        for pad in zeropads:
            general_params.append(("ZeroPadding2D", ((pad[0], pad[1]), (pad[2], pad[3]))))
        if self.model_output_shape == 1:
            for drp in dropouts:
                general_params.append(("end", drp, 1, "sigmoid"))
        else:
            for drp in dropouts:
                general_params.append(("end", drp, self.model_output_shape, "softmax"))
        general_count = range(dw_count[-1]+1, dw_count[-1] + len(general_params)+1)
        general_token = dict(zip(general_count, general_params))

        tokens = {**c2d_token, **dw_token, **general_token}
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
        model.add(keras.layers.InputLayer(input_shape=model_input_shape))

        for layer in layers_info:
            if layer[0] == "Conv2D":
                model.add(keras.layers.Conv2D(filters=layer[1], kernel_size=layer[2], strides=layer[3],
                                              padding=layer[4], activation="linear"))
                model.add(keras.layers.BatchNormalization(axis=[3]))
                model.add(keras.layers.ReLU(max_value=6))

            elif layer[0] == "DepthwiseConv2D":
                model.add(keras.layers.DepthwiseConv2D(kernel_size=layer[1], strides=layer[2], padding=layer[3],
                                                       activation="linear"))
                model.add(keras.layers.BatchNormalization(axis=[3]))
                model.add(keras.layers.ReLU(max_value=6))

            elif layer[0] == "ZeroPadding2D":
                model.add(keras.layers.ZeroPadding2D(padding=layer[1]))

            elif layer[0] == "end":
                model.add(keras.layers.GlobalAvgPool2D(data_format="channels_last"))
                model.add(keras.layers.Reshape(target_shape=(1, 1, model.layers[-1].output_shape[1])))
                model.add(keras.layers.Dropout(rate=layer[1]))
                model.add(keras.layers.Conv2D(filters=layer[2], kernel_size=(1, 1), strides=(1, 1), padding="same",
                                              activation="linear"))
                model.add(keras.layers.Reshape(target_shape=(layer[2],)))
                model.add(keras.layers.Activation(layer[3]))

            else:
                raise ValueError(f"emnas: Layer Type Unknown {str(layer)}")

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
                print("Skipped:", sequence, "due to:", e)
                architectures.append(None)

        return architectures

    def check_sequence(self, sequence: List) -> bool:
        tokens = self.generate_token()
        zero_pad_tokens = [x for x, y in tokens.items() if "ZeroPadding2D" in y]
        end_tokens = [x for x, y in tokens.items() if "end" in y]

        for i, token in enumerate(sequence):
            if i == 0 and (token in zero_pad_tokens or token in end_tokens):
                return False
            if i != len(sequence) - 1 and token in end_tokens:
                return False
            if i == len(sequence) - 1 and token not in end_tokens:
                return False

        if not len(sequence):
            return False
        return True
