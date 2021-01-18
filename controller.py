"""
controller
generates sequences of token keys, based on lstm predictor
"""
import keras
import numpy as np


class Controller(object):

    def __init__(self, search_space_length):
        self.no_of_samples = 10
        self.max_no_of_layers = 4
        self.controller_rnn_dim = 100
        self.controller_classes = search_space_length

    def generate_sequence(self, tokens):
        rnn = self.controller_rnn()
        token_keys = list(tokens.keys())
        dense_tokens = [x for x, y in tokens.items() if "Dense" in y]

        samples = []
        i = 0
        while i < self.no_of_samples:
            sequence = np.zeros((1, 1, self.max_no_of_layers))  # start with zeros
            dense_flag = False
            j = 0
            while j < self.max_no_of_layers-1:
                predictions = rnn.predict(sequence)[0][0]
                selected = np.random.choice(token_keys, p=predictions)
                if selected == token_keys[-1]:
                    break  # finish the sequence if out_layer is predicted
                if selected in dense_tokens:
                    dense_flag = True
                if dense_flag and selected not in dense_tokens:
                    continue  # avoid conv layer after dense layer
                sequence[0][0][j] = selected
                j += 1

            sequence[0][0][self.max_no_of_layers - 1] = token_keys[-1]  # last layer: out layer
            sequence = sequence[sequence != 0][np.newaxis][np.newaxis]  # drop zeros if terminated earlier
            sequence = sequence.astype(int)[0][0].tolist()
            if sequence in samples:
                continue  # no repeated sequence in samples
            samples.append(sequence)
            i += 1
        return samples

    def controller_rnn(self):
        controller_input_shape = (None, self.max_no_of_layers)
        main_input = keras.engine.input_layer.Input(shape=controller_input_shape, batch_shape=None, name='main_input')
        x = keras.layers.LSTM(self.controller_rnn_dim, return_sequences=True)(main_input)
        main_output = keras.layers.Dense(self.controller_classes, activation='softmax', name='main_output')(x)
        model = keras.models.Model(inputs=[main_input], outputs=[main_output])
        return model



