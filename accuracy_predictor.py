import pandas as pd
import os
import config
import time
from trainer import Trainer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras


latency_dataset = "latency_datasets/Dataset_4"
table_name = "table"
trainer_ = Trainer()

# class AccuracyPredictor(object):
#     def __init__(self):
#         self.latency_dataset = "latency_datasets/Dataset_3"
#         self.trainer = Trainer()


def measure_accuracy():
    df = pd.read_csv(f"{latency_dataset}/{table_name}.csv")
    h5_list = [i for i in os.listdir(latency_dataset) if ".h5" in i]
    h5_list.sort()
    for i, h5 in enumerate(h5_list):
        t1 = time.time()
        h5_name = h5.split(".")[0]
        h5_index = df[df["model"] == h5_name].index
        current_cell = df.loc[h5_index[0], "accuracy"]

        if not pd.isna(current_cell):
            print(h5_name, "already measured:", current_cell)
            continue

        samples = [[None]]  # latency prediction disabled
        architectures = [keras.models.load_model(f"{latency_dataset}/{h5}")]
        try:
            epoch_performance = trainer_.train_models(samples=samples, architectures=architectures)
            accuracy = list(epoch_performance.values())[0][0]
        except Exception as e:
            print(h5_name, "Accuracy measurement failed.")
            print(e)
            continue

        df.at[h5_index, "accuracy"] = round(accuracy, 2)
        print(h5_name, "Accuracy measurement done.", round(time.time()-t1, 2), "sec")

        if (i+1) % 10 == 0:
            print("Saving ..")
            df.to_csv(f"{latency_dataset}/{table_name}.csv", index=False)  # save to hdd every 10 iter.

    df.to_csv(f"{latency_dataset}/{table_name}.csv", index=False)


if __name__ == '__main__':
    measure_accuracy()
    print("DONE")














