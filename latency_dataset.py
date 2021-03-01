import pandas as pd
import config
import time
import random
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
from camera_drive import SipeedCamera
from controller import Controller
from search_space import SearchSpace


no_of_examples = 10
kmodel_limit = 3847*1024
latency_dataset = "latency_datasets/Dataset_1"
model_input_shape = config.emnas["model_input_shape"]


def generate_models():
    if os.listdir(latency_dataset):
        raise ValueError("Dataset folder is not empty.")
    search_space = SearchSpace(model_output_shape=2)
    tokens = search_space.generate_token()
    controller = Controller(tokens=tokens)
    architectures = []
    df = pd.DataFrame(columns=["model", "params", "sipeed_latency [ms]", "kmodel_memory [B]", "cpu_latency [ms]",
                               "model_info"])

    i = 0
    while i < no_of_examples:
        sequence = controller.generate_sequence_naive(mode="r") + [list(tokens.keys())[-1]]
        if (sequence in architectures) or (not controller.check_sequence(sequence)):
            continue
        architecture = search_space.create_model(sequence=sequence, model_input_shape=model_input_shape)
        architectures.append(architecture)
        i += 1
        file_name = f"model_{i}"
        architecture.save(f"{latency_dataset}/{file_name}.h5")
        model_params = architecture.count_params()
        model_info = search_space.translate_sequence(sequence)
        df = df.append({"model": file_name, "params": model_params, "model_info": model_info}, ignore_index=True)

    df.to_csv(f"{latency_dataset}/table.csv", index=False)


def measure_sipeed_latency():
    df = pd.read_csv(f"{latency_dataset}/table.csv")
    kmodel_list = [i for i in os.listdir(latency_dataset) if ".kmodel" in i]
    for kmodel in kmodel_list:
        t1 = time.time()
        kmodel_name = kmodel.split(".")[0]
        kmodel_index = df[df["model"] == kmodel_name].index
        kmodel_memory = df.loc[kmodel_index, "kmodel_memory [B]"].iloc[0]

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
    sipeed_cam = SipeedCamera()
    t0 = time.time()

    # Step 1
    # generate_models()
    # print("Model generation done.", time.time() - t0)

    # Step 2
    # sipeed_cam.convert_kmodel(latency_dataset)
    # print("Kmodel conversion done.", time.time() - t0)

    # Step 3
    # manually copy kmodel files to camera
    # measure_sipeed_latency()
    # print("Sipeed cam latency measurement done.", time.time() - t0)

    # Step 4
    # measure_cpu_latency()
    # print("CPU latency measurement done.", time.time() - t0)
