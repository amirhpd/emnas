import keras
import numpy as np
from controller import Controller
# from emnas import plot_image, save_logs
from search_space import SearchSpace
from search_space_mn import SearchSpaceMn
from trainer import Trainer
from camera_drive import CameraDrive
import tensorflow as tf


def test_search_space():
    search_space = SearchSpace(model_output_shape=2)
    token = search_space.generate_token()

    dense_tokens = [x for x, y in token.items() if "Dense" in y]  # dense layers start from 865
    sample_sequence = [52, 146, 31, 119, 138, 244]
    translated_sequence = search_space.translate_sequence(sample_sequence)
    assert len(translated_sequence) == 4

    model = search_space.create_model(sequence=sample_sequence, model_input_shape=(128, 128, 3))
    keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
    print(model.summary())
    assert len(token) == 890


def test_trainer():
    search_space = SearchSpace(model_output_shape=2)
    tokens = search_space.generate_token()
    # controller = Controller(tokens=tokens)
    trainer = Trainer()

    # samples = controller.generate_sequence()
    samples = [[65, 146, 143, 201, 281, 382]]
    architectures = search_space.create_models(samples=samples, model_input_shape=(128, 128, 3))
    epoch_performance = trainer.train_models(samples=samples, architectures=architectures)
    assert len(epoch_performance) != 0


def test_controller_rnn_trainer():
    search_space = SearchSpace(model_output_shape=2)
    tokens = search_space.generate_token()
    controller = Controller(tokens=tokens)
    # samples = controller.generate_sequence()
    manual_epoch_performance = {
        (320, 96, 338, 84, 176, 382): (0.968, 0),  # (acc, lat)
        (22, 47, 225, 315, 223, 382): (0.87, 0),
        (74, 204, 73, 236, 309, 382): (0.74, 0),
        (110, 60, 191, 270, 199, 382): (0.51, 0)
    }

    loss_avg = controller.train_controller_rnn(epoch_performance=manual_epoch_performance)
    print(loss_avg)


def test_controller_sample_generator():
    search_space = SearchSpace(model_output_shape=2)
    tokens = search_space.generate_token()
    controller = Controller(tokens=tokens)
    samples = controller.generate_sequence()
    print(samples)


def test_controller_generate_sequence_naive():
    search_space = SearchSpace(model_output_shape=2)
    tokens = search_space.generate_token()
    controller = Controller(tokens=tokens)

    # samples = controller.generate_sequence_naive(mode="b")
    # for sequence in samples:
    #     sequence_ = sequence
    #     print(sequence_)

    # sequences_random = controller.generate_sequence_naive(mode="r")

    for i in range(20):
        sequences_random = controller.generate_sequence_naive(mode="r_var_len")
        print(sequences_random)
    print("Done.")


def test_sipeed_get_latency():
    sipeed_cam = CameraDrive()
    latency = sipeed_cam.get_latency(model_file="model_0001.kmodel")
    print(latency)


def test_convert_kmodel():
    latency_dataset = "/home/amirhossein/Codes/NAS/mobileNet/converted"
    sipeed_cam = CameraDrive()
    sipeed_cam.convert_kmodel(latency_dataset)


def test_search_space_mobilenets():
    search_space = SearchSpaceMn(model_output_shape=2)
    # token = search_space.generate_token()

    # sample_sequence = [1, 25, 29, 34]
    # mobnet_sequence = [4, 26, 5, 29, 27, 1, 26, 9, 29, 27, 13, 26, 13, 29, 27, 17, 26, 17, 26, 17, 26, 17, 26, 17, 26,
    #                    17, 29, 27, 21, 26, 21, 34]
    mobnet_sequence = [4, 26, 5, 43, 31, 9, 26, 9, 43, 31, 13, 26, 13, 43, 31, 17, 26, 17, 26, 17, 26, 17, 26, 17, 26,
                       17, 43, 31, 21, 26, 21, 49]
    valid_sequence = search_space.check_sequence(mobnet_sequence)
    translated_sequence = search_space.translate_sequence(mobnet_sequence)

    # model = search_space.create_model(sequence=mobnet_sequence, model_input_shape=(128, 128, 3))
    model = search_space.create_models(samples=[mobnet_sequence], model_input_shape=(128, 128, 3))[0]
    # keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
    print(model.summary())


def test_create_convert_manual_sequence():
    search_space = SearchSpaceMn(model_output_shape=2)
    mobnet_sequence = [4, 26, 5, 43, 31, 9, 26, 9, 43, 31, 13, 26, 13, 43, 31, 17, 26, 17, 26, 17, 26, 17, 26, 17, 26,
                       17, 43, 31, 21, 26, 21, 49]
    valid_sequence = search_space.check_sequence(mobnet_sequence)
    assert valid_sequence is True
    # model = search_space.create_models(samples=[mobnet_sequence], model_input_shape=(128, 128, 3))[0]
    model = search_space.create_model(sequence=mobnet_sequence, model_input_shape=(128, 128, 3))
    model.save("manual_model.h5")
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file("manual_model.h5")
    tflite_model = converter.convert()
    open("manual_model.tflite", "wb").write(tflite_model)


# ./nncase/ncc compile latency_datasets/Dataset_/model_0001.tflite latency_datasets/Dataset_/model_0001.kmodel -i
# tflite -o kmodel --dataset nncase/calibration_dataset


def test_multi_objective():
    search_space = SearchSpaceMn(model_output_shape=2)
    tokens = search_space.generate_token()
    trainer = Trainer(tokens)
    max_no_of_layers = 32
    sequence = [4, 26, 5, 43, 31, 9, 26, 9, 43, 31, 13, 26, 13, 43, 31, 17, 26, 17, 26, 17, 26, 17, 26, 17, 26,
                17, 43, 31, 21, 26, 21]  # sequence is without end token
    # sequence = [4, 26, 5, 43, 31, 9, 26, 9, 43, 31, 13, 26, 13, 43, 31, 17, 26, 17, 26, 17, 26, 17, 26, 17, 26]
    if len(sequence) < max_no_of_layers - 1:
        for _ in range((max_no_of_layers - 1) - len(sequence)):
            sequence.append(0)
    acc, lat = trainer.performance_estimate(sequence)
    reward = trainer.multi_objective_reward(acc, lat)
    print("accuracy:", acc, "latency:", lat, "reward:", reward)
