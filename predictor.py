import os
import json
import keras
import config
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib
import cv2

matplotlib.use('TkAgg')
plt.rcParams.update({'font.size': 20, 'mathtext.fontset': 'stix', 'font.family': 'STIXGeneral'})


class Predictor(object):

    def __init__(self):
        self.prediction_dataset = config.predictor["prediction_dataset"]
        self.search_space_len = config.predictor["search_space_len"]
        self.epochs = config.predictor["no_of_epochs"]
        self.mode_invalids = config.predictor["mode_invalids"]  # fill, ignore
        self.mode_predictor = config.predictor["mode_predictor"]  # latency, accuracy
        self.model = None
        self.longest_len = 0
        self.label_column = "sipeed_latency [ms]" if self.mode_predictor == "latency" else "accuracy"
        self.bad_reward = 1000 if self.mode_predictor == "latency" else 0.4

    def pre_process_dataset(self):
        df_in = pd.read_csv(f"{self.prediction_dataset}/table.csv")
        df_in[self.label_column] = df_in[self.label_column].fillna(0)

        if self.mode_invalids == "fill":
            df_in[self.label_column] = df_in[self.label_column].apply(lambda i: self.bad_reward if i == 0 else i)
        if self.mode_invalids == "ignore":
            df_in = df_in[df_in[self.label_column] != 0]

        df = df_in
        df["token_sequence"] = df["token_sequence"].apply(lambda i: json.loads(i))

        self.longest_len = max([len(i) for i in list(df["token_sequence"].values)])  # add zero to make lengths equal
        for i in df["token_sequence"].values:
            for _ in range(self.longest_len - len(i)):
                i.append(0)

        df_tokens = pd.DataFrame.from_dict(dict(zip(df["token_sequence"].index, df["token_sequence"].values))).T
        df_tokens.columns = [f"layer_{i}" for i in range(1, len(df_tokens.columns) + 1)]
        df_tokens[self.label_column] = df[self.label_column]
        df = df_tokens

        x = df.loc[:, df.columns != self.label_column]
        y = df.loc[:, df.columns == self.label_column]

        x_hot = keras.utils.to_categorical(x, num_classes=self.search_space_len + 1)
        x_train, x_test, y_train, y_test = train_test_split(x_hot, y, test_size=0.1, random_state=42)
        return x_train, x_test, y_train, y_test

    def nn_model(self):
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(64, input_shape=(self.longest_len, self.search_space_len + 1),
                                          activation='relu'))
        self.model.add(keras.layers.Dense(64, activation='relu'))
        self.model.add(keras.layers.Dense(32, activation='relu'))
        self.model.add(keras.layers.Dense(32, activation='relu'))
        self.model.add(keras.layers.Dense(32, activation='relu'))
        self.model.add(keras.layers.Dense(16, activation='relu'))
        self.model.add(keras.layers.Dense(16, activation='relu'))
        self.model.add(keras.layers.Dense(8, activation='relu'))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(8, activation='relu'))
        self.model.add(keras.layers.Dense(16, activation='relu'))
        self.model.add(keras.layers.Dense(16, activation='relu'))
        self.model.add(keras.layers.Dense(32, activation='relu'))
        self.model.add(keras.layers.Dense(32, activation='relu'))
        self.model.add(keras.layers.Dense(32, activation='relu'))
        self.model.add(keras.layers.Dense(64, activation='relu'))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Dense(1, activation='linear'))

        optimizer = keras.optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return self.model

    def train_validate_predictor(self, x_train, y_train, x_test, y_test):
        history = self.model.fit(x_train, y_train, epochs=self.epochs, validation_split=0.1, verbose=3)

        hist_loss = history.history["loss"]
        hist_mae = history.history["mae"]
        hist_val_loss = history.history["val_loss"]
        hist_val_mae = history.history["val_mae"]

        fig = plt.figure(figsize=(24, 6))
        plt.plot(np.arange(len(hist_mae)), hist_mae, label="Train MEA")
        plt.plot(np.arange(len(hist_val_mae)), hist_val_mae, label="Validation MAE")
        plt.title("Mean Absolute Error")
        plt.xlabel("Epochs")
        plt.ylabel("MAE")
        plt.grid()
        plt.legend()
        plt.savefig("predictors/fig_1.png")

        fig = plt.figure(figsize=(24, 6))
        plt.plot(np.arange(len(hist_loss)), hist_loss, label="Train Loss")
        plt.plot(np.arange(len(hist_val_loss)), hist_val_loss, label="Validation Loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid()
        plt.legend()
        plt.savefig("predictors/fig_2.png")

        mse_nn, mae_nn = self.model.evaluate(x_test, y_test, verbose=0)
        print("mse: ", mse_nn)
        print("mae: ", mae_nn)
        y_predicted = self.model.predict(x_test)

        width = 0.2
        fig = plt.figure(figsize=(24, 10))
        plt.bar(np.arange(len(y_test)) - width / 2, y_test[self.label_column].to_list(), width, label="True Values")
        plt.bar(np.arange(len(y_test)) + width / 2, y_predicted.flatten(), width, label="Predicted Values")
        plt.title("Validation Set Comparison")
        plt.xlabel("Examples")
        plt.ylabel(self.mode_predictor)
        plt.grid()
        plt.legend()
        plt.savefig("predictors/fig_3.png")

        img_list = [cv2.imread(f"predictors/fig_{i}.png") for i in range(1, 4)]
        img = cv2.vconcat(img_list)
        cv2.imwrite(f"predictors/fig_{self.mode_predictor}_predictor.png", img)
        [os.remove(f"predictors/fig_{i}.png") for i in range(1, 4)]


if __name__ == '__main__':
    predictor = Predictor()
    train_x, test_x, train_y, test_y = predictor.pre_process_dataset()
    predictor.nn_model()
    predictor.train_validate_predictor(train_x, train_y, test_x, test_y)
    predictor.model.save(f"predictors/{predictor.mode_predictor}_predictor.h5")
    # os.system("xdg-open fig_predictor.png")
