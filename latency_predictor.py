import pandas as pd
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras


class LatencyPredictor(object):
    def __init__(self):
        self.latency_dataset = "misc"
        self.outlier_limit = {"Sipeed": 2000, "Jevois": 100}
        self.lr = 0.001
        self.train_epochs = 1000

    def process_dataset(self):
        df = pd.read_csv(f"{self.latency_dataset}/table.csv")
        df["token_sequence"] = df["token_sequence"].apply(lambda x: json.loads(x))
        df_tokens = pd.DataFrame.from_dict(dict(zip(df["token_sequence"].index, df["token_sequence"].values))).T
        df_tokens.columns = [f"layer_{i}" for i in range(1, len(df_tokens.columns)+1)]
        df = pd.concat([df, df_tokens], axis=1)
        df = df.drop(["token_sequence", "model_info"], axis=1)

        df = df.dropna()
        df = df.drop(["params [K]", "model", "kmodel_memory [KB]",
                      "cpu_latency [ms]", "layer_6"], axis=1)
        return df

    def pre_process(self):
        df = self.process_dataset()
        df = df[df["jevois_latency [ms]"] <= self.outlier_limit["Jevois"]]  # remove outliers
        df = df[df["sipeed_latency [ms]"] <= self.outlier_limit["Sipeed"]]  # remove outliers

        x = df.drop(["jevois_latency [ms]", "sipeed_latency [ms]"], axis=1)
        x = keras.utils.to_categorical(x, num_classes=383)
        y_jevois = df.loc[:, df.columns == "jevois_latency [ms]"].round(0)
        y_sipeed = df.loc[:, df.columns == "sipeed_latency [ms]"].round(0)

        return x, y_jevois, y_sipeed

    def regressor(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(64, input_shape=(5, 383), activation='relu'))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(8, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(8, activation='relu'))
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(1, activation='linear'))

        optimizer = keras.optimizers.Adam(lr=self.lr)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    def train(self):
        x, y_jevois, y_sipeed = self.pre_process()

        model_j = self.regressor()
        history_j = model_j.fit(x, y_jevois, epochs=self.train_epochs, validation_split=0.1, verbose=3)
        val_loss_j = history_j.history["val_loss"][-1]
        val_mae_j = history_j.history["val_mae"][-1]
        model_j.save(f"{self.latency_dataset}/regressor_jevois.h5")

        model_s = self.regressor()
        history_s = model_s.fit(x, y_sipeed, epochs=self.train_epochs, validation_split=0.1, verbose=3)
        val_loss_s = history_s.history["val_loss"][-1]
        val_mae_s = history_s.history["val_mae"][-1]
        model_s.save(f"{self.latency_dataset}/regressor_sipeed.h5")

        history = {
            "Jevois": {"val_loss": val_loss_j, "val_mae": val_mae_j},
            "Sipeed": {"val_loss": val_loss_s, "val_mae": val_mae_s},
        }
        return history

    def inference(self, sequence, hardware):
        model = keras.models.load_model(f"{self.latency_dataset}/regressor_{hardware}.h5")
        data = {
            "layer_1": sequence[0],
            "layer_2": sequence[1],
            "layer_3": sequence[2],
            "layer_4": sequence[3],
            "layer_5": sequence[4],
        }
        x_predict = pd.DataFrame(data, index=[0])
        x_predict_hot = keras.utils.to_categorical(x_predict, num_classes=383)
        prediction = model.predict(x_predict_hot)[0][0]
        return prediction

















