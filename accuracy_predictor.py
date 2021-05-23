import pandas as pd
import os
import config
import time
from trainer import Trainer
from search_space import SearchSpace
from search_space_mn import SearchSpaceMn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras


latency_dataset = "latency_datasets/Dataset_6"
table_name = "table"
model_output_shape = config.emnas["model_output_shape"]
if config.search_space["mode"] == "MobileNets":
    search_space = SearchSpaceMn(model_output_shape=model_output_shape)
else:
    search_space = SearchSpace(model_output_shape=model_output_shape)
tokens = search_space.generate_token()
trainer = Trainer(tokens)


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
        if pd.isna(df.loc[h5_index[0], "sipeed_latency [ms]"]):
            print(h5_name, "skipped")
            continue

        architectures = [keras.models.load_model(f"{latency_dataset}/{h5}")]
        try:
            accuracy = trainer.train_models(architectures=architectures)[0]
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














