import pandas as pd
import config
import time
import random
import numpy as np
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
from camera_drive import SipeedCamera
from controller import Controller
from search_space import SearchSpace


no_of_examples = 1000
kmodel_limit = 3847
latency_dataset = "latency_datasets/Dataset_3"
model_input_shape = config.emnas["model_input_shape"]


def generate_models():
    if os.listdir(latency_dataset):
        raise ValueError("Dataset folder is not empty.")
    search_space = SearchSpace(model_output_shape=2)
    tokens = search_space.generate_token()
    controller = Controller(tokens=tokens)
    architectures = []
    df = pd.DataFrame(columns=["model", "params [K]", "sipeed_latency [ms]", "kmodel_memory [KB]", "cpu_latency [ms]",
                               "token_sequence", "model_info"])

    i = 0
    while i < no_of_examples:
        sequence = controller.generate_sequence_naive(mode="r") + [list(tokens.keys())[-1]]
        if (sequence in architectures) or (not controller.check_sequence(sequence)):
            continue
        try:
            architecture = search_space.create_model(sequence=sequence, model_input_shape=model_input_shape)
        except Exception as e:
            print(sequence)
            print(e)
            continue
        architectures.append(architecture)
        i += 1
        i_str = format(i, f"0{len(str(no_of_examples))}d")  # add 0s
        file_name = f"model_{i_str}"
        architecture.save(f"{latency_dataset}/{file_name}.h5")
        model_params = round(architecture.count_params()/1000, 4)
        model_info = search_space.translate_sequence(sequence)
        model_info_json = json.dumps(dict(zip(range(len(model_info)), model_info)))
        df = df.append({"model": file_name, "params [K]": model_params, "token_sequence": sequence,
                        "model_info": model_info_json}, ignore_index=True)

    df.to_csv(f"{latency_dataset}/table.csv", index=False)


def measure_sipeed_latency():
    df = pd.read_csv(f"{latency_dataset}/table.csv")
    kmodel_list = [i for i in os.listdir(latency_dataset) if ".kmodel" in i]
    kmodel_list.sort()
    for i, kmodel in enumerate(kmodel_list):
        t1 = time.time()
        kmodel_name = kmodel.split(".")[0]
        kmodel_index = df[df["model"] == kmodel_name].index
        kmodel_memory = df.loc[kmodel_index, "kmodel_memory [KB]"].iloc[0]

        current_cell = df.loc[kmodel_index[0], "sipeed_latency [ms]"]
        if not pd.isna(current_cell):
            print(kmodel_name, "already measured:", current_cell)
            continue

        if kmodel_memory < kmodel_limit:
            try:
                latency = sipeed_cam.get_latency(model_file=kmodel)
            except Exception as e:
                print(kmodel_name, "Latency measurement failed.")
                print(e)
                continue
        else:
            print(kmodel_name, "Too large.")
            latency = pd.np.nan

        df.at[kmodel_index, "sipeed_latency [ms]"] = round(latency, 2)
        print(kmodel_name, "Latency measurement on Sipeed done.", time.time()-t1, "sec")

        if (i+1) % 10 == 0:
            print("Saving ..")
            df.to_csv(f"{latency_dataset}/table.csv", index=False)  # save to hdd every 10 iter.

    df.to_csv(f"{latency_dataset}/table.csv", index=False)


def _get_test_images(no_of_images):
    image_list = []
    for path, subdirs, files in os.walk("nncase/calibration_dataset"):
        for name in files:
            image_list.append((os.path.join(path, name)))

    selected = random.choices(image_list, k=no_of_images)
    return selected


def measure_cpu_latency():
    df = pd.read_csv(f"{latency_dataset}/table.csv")
    h5_list = [i for i in os.listdir(latency_dataset) if ".h5" in i]
    h5_list.sort()
    images = _get_test_images(20)

    for h5 in h5_list:
        model_name = h5.split(".")[0]
        model_index = df[df["model"] == model_name].index
        model = keras.models.load_model(f"{latency_dataset}/{h5}")
        latency_ = []
        for img in images:
            test_image = keras.preprocessing.image.load_img(img, target_size=(128, 128))
            test_image = keras.preprocessing.image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            t1 = time.time()
            _ = model.predict(test_image)
            latency_.append((time.time() - t1)*1000)
        latency_h5 = sum(latency_) / len(latency_)
        df.at[model_index, "cpu_latency [ms]"] = round(latency_h5, 2)
        print(model_name, "Latency measurement on CPU done.")

    df.to_csv(f"{latency_dataset}/table.csv", index=False)


if __name__ == '__main__':
    step = 2
    sipeed_cam = SipeedCamera()

    if step == 1:
        # Step 1.1
        print("Generating models ..")
        t0 = time.time()
        generate_models()
        print("Model generation done.", round(time.time()-t0, 2), "s")

        # Step 1.2
        print("Converting models ..")
        t0 = time.time()
        sipeed_cam.convert_kmodel(latency_dataset)
        print("Kmodel conversion done.", round(time.time()-t0, 2), "s")

        # Step 1.3
        print("Measuring CPU latency ..")
        t0 = time.time()
        measure_cpu_latency()
        print("CPU latency measurement done.", round(time.time()-t0, 2), "s")

    if step == 2:
        # Step 2
        # manually copy kmodel files to camera
        print("Measuring Sipeed latency ..")
        t0 = time.time()
        measure_sipeed_latency()
        print("Sipeed cam latency measurement done.", round(time.time()-t0, 2), "s")
