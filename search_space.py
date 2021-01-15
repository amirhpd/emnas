"""
search_space
creates different possibilities of NN in form of mapping
converts a mapping to keras model
"""
import os
import warnings
from typing import Dict, List, Tuple
import pandas as pd
from keras import optimizers
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout


class SearchSpace(object):

    def __init__(self, target_classes):

        self.target_classes = target_classes
        self.mapping = self.create_mapping()
        self.architecture_dropout = 0.2
        self.architecture_loss_function = "categorical_crossentropy"
        self.architecture_optimizer = "Adam"
        self.architecture_lr = 0.01
        self.architecture_decay = 0.0
        self.metrics = ["accuracy"]

    def create_mapping_cnn(self):
        pass

    def create_mapping(self) -> Dict[int, Tuple[int, str]]:
        nodes = [8, 16, 32, 64, 128, 256, 512]
        act_funcs = ['sigmoid', 'tanh', 'relu', 'elu']
        layer_params = []
        layer_id = []
        for i in range(len(nodes)):
            for j in range(len(act_funcs)):
                layer_params.append((nodes[i], act_funcs[j]))
                layer_id.append(len(act_funcs) * i + j + 1)
        vocab = dict(zip(layer_id, layer_params))
        vocab[len(vocab) + 1] = (('dropout'))
        if self.target_classes == 2:
            vocab[len(vocab) + 1] = (self.target_classes - 1, 'sigmoid')
        else:
            vocab[len(vocab) + 1] = (self.target_classes, 'softmax')
        return vocab

    def encode_sequence(self, sequence: List[Tuple[int, str]]) -> List[int]:
        keys = list(self.mapping.keys())
        values = list(self.mapping.values())
        encoded_sequence = []
        for value in sequence:
            encoded_sequence.append(keys[values.index(value)])
        return encoded_sequence

    def decode_sequence(self, sequence: Dict[int, Tuple[int, str]]) -> List[Tuple[int, str]]:
        keys = list(self.mapping.keys())
        values = list(self.mapping.values())
        decoded_sequence = []
        for key in sequence:
            decoded_sequence.append(values[keys.index(key)])
        return decoded_sequence

    def create_architecture(self, sequence: List[int], mlp_input_shape: Tuple):
        layer_configs = self.decode_sequence(sequence)
        model = Sequential()
        if len(mlp_input_shape) > 1:
            model.add(Flatten(name='flatten', input_shape=mlp_input_shape))
            for i, layer_conf in enumerate(layer_configs):
                if layer_conf is 'dropout':
                    model.add(Dropout(self.architecture_dropout, name='dropout'))
                else:
                    model.add(Dense(units=layer_conf[0], activation=layer_conf[1]))
        else:
            for i, layer_conf in enumerate(layer_configs):
                if i == 0:
                    model.add(Dense(units=layer_conf[0], activation=layer_conf[1], input_shape=mlp_input_shape))
                elif layer_conf is 'dropout':
                    model.add(Dropout(self.architecture_dropout, name='dropout'))
                else:
                    model.add(Dense(units=layer_conf[0], activation=layer_conf[1]))

        if self.target_classes == 2:
            self.architecture_loss_function = "binary_crossentropy"
        optimizer = getattr(optimizers, self.architecture_optimizer)(lr=self.architecture_lr,
                                                                     decay=self.architecture_decay)
        model.compile(loss=self.architecture_loss_function, optimizer=optimizer, metrics=self.metrics)

        return model
