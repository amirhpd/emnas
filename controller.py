"""
controller
generates sequences of token keys, based on lstm predictor
"""
import os
import keras
import numpy as np


class Controller(object):

    def __init__(self, search_space_length):
        self.no_of_samples = 10
        self.max_no_of_layers = 4
        self.rnn_dim = 100
        self.rnn_classes = search_space_length
        self.rnn_optimizer = "Adam"
        self.rnn_lr = 0.01
        self.rnn_decay = 0.1
        self.rnn_weights = 'logs/rnn_weights.h5'
        self.rnn_no_of_epochs = 10
        self.data = []
        self.samples = []
        self.rnn_model = self.controller_rnn()

    def generate_sequence(self, tokens):
        rnn = self.controller_rnn()
        token_keys = list(tokens.keys())
        dense_tokens = [x for x, y in tokens.items() if "Dense" in y]

        i = 0
        while i < self.no_of_samples:
            sequence = np.zeros((1, 1, self.max_no_of_layers))  # start with zeros
            dense_flag = False
            j = 0
            while j < self.max_no_of_layers-1:
                predictions = rnn.predict(sequence[:,:,:-1])[0][0]  # predict on layers, leave out_layer for rnn_y
                selected = np.random.choice(token_keys, p=predictions)
                if selected == token_keys[-1]:
                    break  # finish the sequence if out_layer is predicted
                if selected in dense_tokens:
                    dense_flag = True
                if dense_flag and selected not in dense_tokens:
                    continue  # avoid conv layer after dense layer
                sequence[0][0][j] = selected
                j += 1

            sequence[0][0][self.max_no_of_layers-1] = token_keys[-1]  # last layer: out layer
            sequence = sequence[sequence != 0][np.newaxis][np.newaxis]  # drop zeros if terminated earlier
            sequence = sequence.astype(int)[0][0].tolist()
            if sequence in self.samples:
                continue  # no repeated sequence in samples
            self.samples.append(sequence)
            i += 1
        return self.samples

    def controller_rnn(self):
        controller_input_shape = (None, self.max_no_of_layers-1)
        main_input = keras.engine.input_layer.Input(shape=controller_input_shape, batch_shape=None, name='main_input')
        x = keras.layers.LSTM(self.rnn_dim, return_sequences=True)(main_input)
        main_output = keras.layers.Dense(self.rnn_classes, activation='softmax', name='main_output')(x)
        model = keras.models.Model(inputs=[main_input], outputs=[main_output])
        return model

    def train_controller_rnn(self, rl_loss_fn):
        rnn_x = np.array(self.samples)[:, :-1]
        rnn_x = rnn_x.reshape(rnn_x.shape[0], 1, rnn_x.shape[1])
        rnn_y = np.array(self.samples)[:, -1]
        rnn_y = keras.utils.to_categorical(rnn_y-1, self.rnn_classes)  # -1: 0 index, but token starts from 1

        optimizer = getattr(keras.optimizers, self.rnn_optimizer)(lr=self.rnn_lr,
                                                                  decay=self.rnn_decay, clipnorm=1.0)
        # self.rnn_model.compile(optimizer=optimizer, loss={'main_output': rl_loss_fn})
        self.rnn_model.compile(optimizer=optimizer, loss="mse")
        if os.path.exists(self.rnn_weights):
            self.rnn_model.load_weights(self.rnn_weights)

        self.rnn_model.fit({'main_input': rnn_x},
                           {'main_output': rnn_y.reshape(len(rnn_y), 1, self.rnn_classes)},
                           epochs=self.rnn_no_of_epochs,
                           batch_size=len(self.data),
                           verbose=0)

        if not os.path.exists(self.rnn_weights):
            os.mknod(self.rnn_weights)
        self.rnn_model.save_weights(self.rnn_weights)

    def reinforce(self):
        pass
