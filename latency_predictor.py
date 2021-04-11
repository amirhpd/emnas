import pandas as pd
import json
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras


class LatencyPredictor(object):
    def __init__(self):
        self.latency_dataset = "latency_datasets/Dataset_3"
        self.mean = None
        self.std = None
        self.model = None
        self.regressor()

    def process_dataset(self):
        df = pd.read_csv(f"{self.latency_dataset}/table.csv")
        df_models = []

        for model_i in range(len(df)):
            model_info = json.loads(df["model_info"][model_i])
            for key, value in model_info.items():
                if len(value) == 3:
                    model_info[key][2:2] = [None]*3
                if len(value) == 2:
                    model_info[key].insert(0, "output")
                    model_info[key][2:2] = [None]*3

            df_model = pd.DataFrame(model_info).T
            df_model.columns = ["layer", "size", "kernel_size", "stride", "padding", "activation"]
            df_models.append(df_model)  # not used

        df["token_sequence"] = df["token_sequence"].apply(lambda x: json.loads(x))
        # df_tokens = pd.DataFrame.from_items(zip(df["token_sequence"].index, df["token_sequence"].values)).T
        df_tokens = pd.DataFrame.from_dict(dict(zip(df["token_sequence"].index, df["token_sequence"].values))).T
        df_tokens.columns = [f"layer_{i}" for i in range(1, len(df_tokens.columns)+1)]
        df_out = pd.concat([df, df_tokens], axis=1)
        df_out = df_out.drop(["token_sequence", "model_info"], axis=1)
        return df_out

    def pre_process(self):
        df = self.process_dataset()
        df = df.dropna()
        df = df.drop(["model", "kmodel_memory [KB]", "cpu_latency [ms]"], axis=1)

        x = df.loc[:, df.columns != "sipeed_latency [ms]"]
        y = df.loc[:, df.columns == "sipeed_latency [ms]"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=123)

        # self.mean = x_train.mean(axis=0)
        # self.std = x_train.std(axis=0)
        # x_train_n = (x_train - self.mean) / self.std
        # x_test_n = (x_test - self.mean) / self.std

        return x_train, x_test, y_train, y_test

    def regressor(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(128, input_shape=(7, ), activation='relu'))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(8, activation='relu'))
        model.add(keras.layers.Dense(1, activation='linear'))

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model = model

    def train(self):
        x_train, x_test, y_train, y_test = self.pre_process()
        history = self.model.fit(x_train, y_train, epochs=100, validation_split=0.01)

        mse_nn, mae_nn = self.model.evaluate(x_test, y_test)
        self.model.save(f"{self.latency_dataset}/regressor.h5")
        return mse_nn, mae_nn

    def inference(self, sequence, architecture):
        model = keras.models.load_model(f"{self.latency_dataset}/regressor.h5")
        data = {
            "params [K]": round(architecture.count_params()/1000, 4),
            "layer_1": sequence[0],
            "layer_2": sequence[1],
            "layer_3": sequence[2],
            "layer_4": sequence[3],
            "layer_5": sequence[4],
            "layer_6": sequence[5],
        }

        prediction = model.predict(pd.DataFrame(data, index=[0]))[0][0]

        # data_n = (pd.Series(data) - self.mean) / self.std
        # prediction = self.model.predict(pd.DataFrame(data_n).T)
        # prediction = (prediction_n * self.std) + self.mean

        return prediction

















