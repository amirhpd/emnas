"""
Search Strategy
generates sequences of token keys, based on lstm predictor
"""
import keras
import numpy as np
import config


class Controller(object):

    def __init__(self, tokens):
        self.no_of_samples_per_epoch = config.controller["no_of_samples_per_epoch"]
        self.no_of_layers = config.controller["no_of_layers"]  # v2 handles sequences equal with length
        self.rnn_dim = config.controller["rnn_dim"]
        self.rnn_lr = config.controller["rnn_lr"]
        self.rnn_decay = config.controller["rnn_decay"]
        self.rnn_no_of_epochs = config.controller["rnn_no_of_epochs"]
        self.rnn_loss_alpha = config.controller["rnn_loss_alpha"]
        self.rl_baseline = config.controller["rl_baseline"]
        self.verbose = config.controller["verbose"]
        self.epoch_performance = None
        self.tokens = tokens
        self.rnn_classes = len(tokens)
        self.rnn_model = self.controller_rnn()

    def generate_sequence(self):
        sequences = []
        token_keys = list(self.tokens.keys())
        dense_tokens = [x for x, y in self.tokens.items() if "Dense" in y]

        sample_seq = 0
        while sample_seq < self.no_of_samples_per_epoch:
            sequence = np.zeros((1, 1, self.no_of_layers - 1), dtype="int32")
            dense_flag = False

            layer = 0
            while layer < self.no_of_layers-1:
                distribution = self.rnn_model.predict(sequence)
                prob = distribution[0][0]
                selected = np.random.choice(token_keys, size=1, p=prob)[0]
                if layer == 0 and (selected in dense_tokens or selected == token_keys[-1] or selected == token_keys[-2]):
                    continue  # no dense, dropout, out_layer on first layer
                if layer != len(sequence)-1 and selected == token_keys[-1]:
                    continue  # no out_layer on middle layers
                if selected in dense_tokens:
                    dense_flag = True
                if dense_flag and selected not in dense_tokens:
                    continue  # avoid conv layer after dense layer
                if not selected == 0:
                    sequence[0][0][layer] = selected
                    layer += 1

            sequence = sequence[0][0].tolist()
            sequence = sequence + [token_keys[-1]]
            if sequence not in sequences:
                sequences.append(sequence)  # no repeated sequence in samples
                sample_seq += 1

        return sequences

    def controller_rnn(self):
        main_input = keras.engine.input_layer.Input(shape=(None, self.no_of_layers - 1),
                                                    batch_shape=None, name="main_input")
        x = keras.layers.LSTM(self.rnn_dim, return_sequences=True)(main_input)
        x = keras.layers.LSTM(self.rnn_dim, return_sequences=True)(x)
        main_output = keras.layers.Dense(self.rnn_classes, activation="softmax", name="main_output")(x)
        model = keras.models.Model(inputs=[main_input], outputs=[main_output])

        model.compile(loss={"main_output": self.reinforce},
                      optimizer=keras.optimizers.Adam(lr=self.rnn_lr, decay=self.rnn_decay, clipnorm=1.0))
        return model

    def train_controller_rnn(self, epoch_performance):
        self.epoch_performance = epoch_performance
        samples = list(epoch_performance.keys())
        rnn_x = np.array(samples)[:, :-1].reshape(len(samples), 1, self.no_of_layers - 1)
        rnn_y = keras.utils.to_categorical(np.array(samples)[:, -1], self.rnn_classes+1).reshape(len(samples),
                                                                                                 1, self.rnn_classes+1)
        history = self.rnn_model.fit({'main_input': rnn_x},
                                     {'main_output': rnn_y},
                                     epochs=self.rnn_no_of_epochs,
                                     batch_size=len(rnn_x),
                                     verbose=self.verbose)
        lstm_loss_avg = np.average(list(history.history.values())[0])
        return lstm_loss_avg

    def reinforce(self, y_true, y_pred):
        rewards = (np.array(list(self.epoch_performance.values())) - self.rl_baseline)[np.newaxis].T
        discounted_rewards = self.discount_reward(rewards)
        y_pred = keras.backend.clip(y_pred, 1e-36, 1e36)
        loss = - keras.backend.log(y_pred) * discounted_rewards[:, None]
        return loss

    def discount_reward(self, rewards):
        discounted_reward = np.zeros_like(rewards, dtype=np.float32)
        for t in range(len(rewards)):
            dis_reward = 0
            for i, r in enumerate(rewards[t:]):
                dis_reward = self.rnn_loss_alpha ** (i - t) * r
            discounted_reward[t] = dis_reward
        if len(rewards) > 1:
            discounted_reward = (discounted_reward - discounted_reward.mean()) / discounted_reward.std()
        return discounted_reward
