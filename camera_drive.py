import os
import serial
import time
import tensorflow as tf
import subprocess
import re
import pandas as pd


class SipeedCamera(object):
    def __init__(self):
        self.connection = None
        self.response = []

    def connect(self):
        try:
            self.connection = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.01)
        except Exception as e:
            try:
                self.connection = serial.Serial('/dev/ttyUSB1', 115200, timeout=0.01)
            except Exception as e:
                raise IOError('Could not connect to Sipeed camera')
        time.sleep(0.5)

    def get_response(self):
        line = self.connection.readline()
        while line != b'':
            line = self.connection.readline()
            self.response.append(line.decode("utf-8"))
        return self.response

    def wait_to_ready(self, timeout):
        t = 0
        while t < timeout:
            t += 1
            r = self.get_response()
            if len(r):
                if r[-2] == ">>> ":
                    return True
            time.sleep(0.01)
        raise OverflowError('Timeout. ">>> " not received.')

    def get_latency(self, model_file):
        self.connect()

        self.connection.write(b'\x03')
        self.wait_to_ready(timeout=500)  # 5s

        self.connection.write(b'import tools')
        self.connection.write(b'\r')
        time.sleep(1)
        self.connection.write(b'import tools')
        self.connection.write(b'\r')
        self.wait_to_ready(timeout=500)

        self.connection.write(str.encode(f'tools.measure_latency("{model_file}")'))
        self.connection.write(b'\r')
        time.sleep(1)
        self.wait_to_ready(timeout=12000)  # 2min

        self.connection.write(b'tools.query_latency()')
        self.connection.write(b'\r')

        response = self.get_response()
        latency = float(response[-3])
        return latency

    def convert_kmodel(self, path):
        h5_list = [i for i in os.listdir(path) if ".h5" in i]
        df = pd.read_csv(f"{path}/table.csv")

        for h5 in h5_list:
            file_path = f"{path}/{h5}".split(".")[0]
            if not os.path.isfile(file_path + ".kmodel"):
                converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(file_path + ".h5")
                tflite_model = converter.convert()
                open(file_path + ".tflite", "wb").write(tflite_model)
                # os.system(f"./nncase/ncc compile {file_path}.tflite {file_path}.kmodel -i tflite -o kmodel "
                #           f"--dataset nncase/calibration_dataset")
                terminal_out = subprocess.run(["./nncase/ncc", "compile", f"{file_path}.tflite", f"{file_path}.kmodel",
                                               "-i", "tflite", "-o", "kmodel", "--dataset",
                                               "nncase/calibration_dataset"],
                                              stdout=subprocess.PIPE)
                pattern = re.compile(r"Working memory usage: (\d+)")  # number after a text
                match = re.findall(pattern, str(terminal_out))
                if len(match):
                    kmodel_memory = int(match[0])
                    kmodel_index = df[df["model"] == h5.split(".")[0]].index
                    df.at[kmodel_index, "kmodel_memory [B]"] = kmodel_memory
                else:
                    print(f"Converting {file_path} failed.")
                print(f"Converting {file_path} done.")

        df.to_csv(f"{path}/table.csv", index=False)
