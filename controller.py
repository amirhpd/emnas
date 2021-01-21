"""
Search Strategy
generates sequences of token keys, based on lstm predictor
"""
import os
import keras
import numpy as np


class Controller(object):

    def __init__(self, tokens):
        self.no_of_samples = 3
        self.max_no_of_layers = 6
        self.rnn_dim = 100
        self.rnn_optimizer = "Adam"
        self.rnn_lr = 0.01
        self.rnn_decay = 0.1
        self.rnn_weights = 'logs/rnn_weights.h5'
        self.rnn_no_of_epochs = 10
        self.rnn_loss_alpha = 0.9
        self.rl_baseline = 0.5
        self.verbose = 1
        self.epoch_performance = []
        self.samples = []
        self.tokens = tokens
        self.rnn_classes = len(tokens)
        self.rnn_model = self.controller_rnn()

    def generate_sequence(self):
        self.samples = []
        token_keys = list(self.tokens.keys())
        dense_tokens = [x for x, y in self.tokens.items() if "Dense" in y]

        i = 0
        while i < self.no_of_samples:
            sequence = np.zeros((1, 1, self.max_no_of_layers))  # start with zeros
            dense_flag = False
            j = 0
            while j < self.max_no_of_layers-1:
                predictions = self.rnn_model.predict(sequence[:,:,:-1])[0][0]  # predict layers, leave out_layer for rnn_y
                selected = np.random.choice(token_keys, p=predictions)
                if j == 0 and (selected in dense_tokens or selected == token_keys[-1] or selected == token_keys[-2]):
                    continue  # no dense, dropout, out_layer on first layer
                if selected == token_keys[-1]:
                    break  # finish the sequence if out_layer is predicted and there is at least one dense
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

    def train_controller_rnn(self, epoch_performance):
        self.epoch_performance = epoch_performance
        # rnn_x = np.array(self.samples)[:, :-1]
        rnn_x = np.array([i+[None]*(max(map(len, self.samples))-len(i)) for i in self.samples])[:, :-1]
        rnn_x = rnn_x.reshape(rnn_x.shape[0], 1, rnn_x.shape[1])
        # rnn_y = np.array(self.samples)[:, -1]
        rnn_y = np.array([i[-1] for i in self.samples])  # to handle sequences smaller than max_length
        rnn_y = keras.utils.to_categorical(rnn_y-1, self.rnn_classes)  # -1: 0 index, but token starts from 1

        optimizer = getattr(keras.optimizers, self.rnn_optimizer)(lr=self.rnn_lr,
                                                                  decay=self.rnn_decay, clipnorm=1.0)
        self.rnn_model.compile(optimizer=optimizer, loss={'main_output': self.reinforce})
        # self.rnn_model.compile(optimizer=optimizer, loss="mse")
        # if os.path.exists(self.rnn_weights):
        #     self.rnn_model.load_weights(self.rnn_weights)

        self.rnn_model.fit({'main_input': rnn_x},
                           {'main_output': rnn_y.reshape(len(rnn_y), 1, self.rnn_classes)},
                           epochs=self.rnn_no_of_epochs,
                           batch_size=len(self.epoch_performance),
                           verbose=self.verbose)

        # if not os.path.exists(self.rnn_weights):
        #     os.mknod(self.rnn_weights)
        # self.rnn_model.save_weights(self.rnn_weights)

    def reinforce(self, target, output):
        rewards = np.array([round(i[1] - self.rl_baseline, 4) for i in self.epoch_performance])[np.newaxis].T
        discounted_rewards = self.get_discounted_reward(rewards)
        loss = - keras.backend.log(output) * discounted_rewards[:, None]
        return loss

    def get_discounted_reward(self, rewards):
        discounted_r = np.zeros_like(rewards, dtype=np.float32)
        for t in range(len(rewards)):
            running_add = 0.
            exp = 0.
            for r in rewards[t:]:
                running_add += self.rnn_loss_alpha**exp * r
                exp += 1
            discounted_r[t] = running_add
        discounted_r = (discounted_r - discounted_r.mean()) / discounted_r.std()
        return discounted_r
