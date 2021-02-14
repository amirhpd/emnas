"""
Search Strategy
A naive controller to generate CNN models based on:
- Random choice
- Brute-force search
"""
import itertools
import numpy as np


class Controller(object):

    def __init__(self, tokens):
        self.mode = "r"  # r: Random, b:brute-force
        self.no_of_samples = 3
        self.max_no_of_layers = 6
        self.rl_baseline = 0.7
        self.tokens = tokens

    def generate_sequence_random_predict(self):
        # Next token is predicted from random distribution
        samples = []
        token_keys = list(self.tokens.keys())
        dense_tokens = [x for x, y in self.tokens.items() if "Dense" in y]

        i = 0
        while i < self.no_of_samples:
            sequence = np.zeros((1, 1, self.max_no_of_layers))  # start with zeros
            dense_flag = False
            j = 0
            while j < self.max_no_of_layers-1:
                selected = np.random.choice(token_keys)
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
            if sequence in samples:
                continue  # no repeated sequence in samples
            samples.append(sequence)
            i += 1
        return samples

    def generate_sequence_naive(self, mode: str):
        if mode == "b":  # Brute-force
            token_keys = list(self.tokens.keys())
            space = itertools.permutations(token_keys, self.max_no_of_layers)
            return space
        if mode == "r":  # Random
            sequence = []
            token_keys = list(self.tokens.keys())
            for i in range(self.max_no_of_layers):
                token = np.random.choice(token_keys)
                sequence.append(token)
            return sequence

    def check_sequence(self, sequence):
        token_keys = list(self.tokens.keys())
        dense_tokens = [x for x, y in self.tokens.items() if "Dense" in y]

        dense_flag = False
        for i, token in enumerate(sequence):
            if i == 0 and (token in dense_tokens or token == token_keys[-1] or token == token_keys[-2]):
                return False
            if i != len(sequence)-1 and token == token_keys[-1]:
                return False
            if i == len(sequence)-1 and token != token_keys[-1]:
                return False
            if token in dense_tokens:
                dense_flag = True
            if dense_flag and i != len(sequence)-1 and token not in dense_tokens:
                return False

        if not len(sequence):
            return False
        return True
