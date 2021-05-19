"""
Search Strategy
generates sequences of token keys, based on lstm predictor
"""
import itertools
import numpy as np
from typing import List
import config
from search_space import SearchSpace
from search_space_mn import SearchSpaceMn
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras


class Controller(object):

    def __init__(self, tokens):
        self.max_no_of_layers = config.controller["max_no_of_layers"]
        self.agent_lr = config.controller["agent_lr"]
        self.min_reward = config.controller["min_reward"]
        self.min_plays = config.controller["min_plays"]
        self.max_plays = config.controller["max_plays"]
        self.alpha = config.controller["alpha"]
        self.gamma = config.controller["gamma"]
        self.model_input_shape = config.emnas["model_input_shape"]
        self.tokens = tokens
        self.len_search_space = len(tokens) + 1
        self.end_token = list(tokens.keys())[-1]
        self.model = self.rl_agent()
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        if config.search_space["mode"] == "MobileNets":
            self.search_space = SearchSpaceMn(config.emnas["model_output_shape"])
        else:
            self.search_space = SearchSpace(config.emnas["model_output_shape"])


    def rl_agent(self):
        model_output_shape = (self.max_no_of_layers - 1, self.len_search_space)
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(512, input_shape=(self.max_no_of_layers - 1,), activation="relu"))
        model.add(keras.layers.Dense(256, activation="relu"))
        model.add(keras.layers.Dense(128, activation="relu"))
        model.add(keras.layers.Dense(64, activation="relu"))
        model.add(keras.layers.Dense(32, activation="relu"))
        model.add(keras.layers.Dense(16, activation="relu"))
        model.add(keras.layers.Dense(16, activation="relu"))
        model.add(keras.layers.Dense(32, activation="relu"))
        model.add(keras.layers.Dense(64, activation="relu"))
        model.add(keras.layers.Dense(128, activation="relu"))
        model.add(keras.layers.Dense(256, activation="relu"))
        model.add(keras.layers.Dense(512, activation="relu"))
        model.add(keras.layers.Dense(model_output_shape[0] * model_output_shape[1], activation="softmax"))
        model.add(keras.layers.Reshape(model_output_shape))

        model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(lr=self.agent_lr))
        return model

    def get_all_action(self, state: np.ndarray) -> (List, np.ndarray, bool):
        true_sequence = False
        actions = []
        distributions = self.model.predict(state)
        for distribution in distributions[0]:
            distribution /= np.sum(distribution)
            action = np.random.choice(self.len_search_space, 1, p=distribution)[0]
            action = 1 if action == 0 else action
            actions.append(int(action))
            if action == self.end_token:
                break

        sequence = actions + [self.end_token] if self.end_token not in actions else actions
        valid_sequence = self.search_space.check_sequence(sequence)
        if valid_sequence:
            valid_model = self.search_space.create_models(samples=[sequence], model_input_shape=self.model_input_shape)
            true_sequence = True if (valid_model[0] is not None and valid_sequence is True) else False

        if len(actions) < self.max_no_of_layers - 1:
            for _ in range((self.max_no_of_layers - 1) - len(actions)):
                actions.append(0)

        return actions, distributions, true_sequence

    def get_valid_action(self, state: np.ndarray) -> (List, np.ndarray, int):
        true_sequence = False
        counter = 0
        while not true_sequence:
            counter += 1
            actions = []
            distributions = self.model.predict(state)
            for distribution in distributions[0]:
                distribution /= np.sum(distribution)
                action = np.random.choice(self.len_search_space, 1, p=distribution)[0]
                action = 1 if action == 0 else action
                actions.append(int(action))
                if action == self.end_token:
                    break

            sequence = actions + [self.end_token] if self.end_token not in actions else actions
            valid_sequence = self.search_space.check_sequence(sequence)
            if valid_sequence:
                valid_model = self.search_space.create_models(samples=[sequence], model_input_shape=self.model_input_shape)
                true_sequence = True if (valid_model[0] is not None and valid_sequence is True) else False

        if len(actions) < self.max_no_of_layers - 1:
            for _ in range((self.max_no_of_layers - 1) - len(actions)):
                actions.append(0)

        return actions, distributions, counter-1

    def remember(self, state, actions, prob, reward):
        model_output_shape = (self.max_no_of_layers - 1, self.len_search_space)
        encoded_action = np.zeros(model_output_shape, np.float32)
        for i, action in enumerate(actions):
            encoded_action[i][action] = 1

        self.gradients.append(encoded_action - prob)
        self.states.append(state)
        self.rewards.append(reward)
        self.probs.append(prob)

    def clear_memory(self):
        self.states.clear()
        self.gradients.clear()
        self.rewards.clear()
        self.probs.clear()

    def get_discounted_rewards(self, rewards_in):
        discounted_rewards = []
        cumulative_total_return = 0

        for reward in rewards_in[::-1]:
            cumulative_total_return = (cumulative_total_return * self.gamma) + reward
            discounted_rewards.insert(0, cumulative_total_return)

        mean_rewards = np.mean(discounted_rewards)
        std_rewards = np.std(discounted_rewards)
        norm_discounted_rewards = (discounted_rewards - mean_rewards) / (std_rewards + 1e-7)

        return norm_discounted_rewards

    def update_policy(self):
        states_ = np.vstack(self.states)

        gradients_ = np.vstack(self.gradients)
        rewards_ = np.vstack(self.rewards)
        discounted_rewards = self.get_discounted_rewards(rewards_)
        discounted_rewards = discounted_rewards.reshape(discounted_rewards.shape[0],
                                                        discounted_rewards.shape[1],
                                                        discounted_rewards.shape[1])
        gradients_ *= discounted_rewards
        gradients_ = self.alpha * gradients_ + np.vstack(self.probs)

        history = self.model.train_on_batch(states_, gradients_)
        self.clear_memory()
        return history

    def generate_sequence_naive(self, mode: str):
        token_keys = list(self.tokens.keys())
        if mode == "b":  # Brute-force
            space = itertools.permutations(token_keys, self.max_no_of_layers - 1)
            return space
        if mode == "r":  # Random
            sequence = []
            sequence_length = np.random.randint(3, self.max_no_of_layers)
            for i in range(sequence_length):
                token = np.random.choice(token_keys)
                sequence.append(token)
            return sequence
        if mode == "r_var_len":
            sequence = []
            length = np.random.randint(3, self.max_no_of_layers - 1, 1)[0]
            for i in range(length):
                token = np.random.choice(token_keys)
                sequence.append(token)
            sequence.append(token_keys[-1])
            return sequence
