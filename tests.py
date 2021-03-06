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
    latency_dataset = ".../NAS/mobileNet/converted"
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
    mobnet_sequence = [6, 31, 15, 48, 25, 7, 21, 28, 13, 12, 25, 35, 35, 43, 24, 29, 49]
    valid_sequence = search_space.check_sequence(mobnet_sequence)
    assert valid_sequence is True
    # model = search_space.create_models(samples=[mobnet_sequence], model_input_shape=(128, 128, 3))[0]
    model = search_space.create_model(sequence=mobnet_sequence, model_input_shape=(128, 128, 3))
    model.save("manual_model.h5")
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file("manual_model.h5")
    tflite_model = converter.convert()
    open("manual_model.tflite", "wb").write(tflite_model)


def test_create_convert_sequences():
    path = ".../NAS/emnas/latency_datasets/Dataset_"
    search_space = SearchSpaceMn(model_output_shape=2)
    sequences = {
        "01": [12, 4, 30, 14, 32, 39, 42, 26, 36, 13, 49],
        "02": [8, 10, 35, 27, 47, 10, 39, 46, 5, 46, 31, 39, 39, 16, 11, 26, 45, 49],
        "03": [10, 10, 10, 9, 2, 44, 25, 19, 1, 3, 36, 49],
        "04": [33, 18, 26, 12, 26, 43, 47, 1, 48, 37, 40, 49],
        "05": [10, 32, 30, 39, 33, 10, 36, 44, 33, 21, 35, 7, 10, 4, 26, 49],
        "06": [41, 8, 9, 43, 25, 34, 33, 9, 43, 35, 48, 49],
    }
    for uid, seq in sequences.items():
        valid_sequence = search_space.check_sequence(seq)
        if not valid_sequence:
            print(f"{uid} failed.")
            continue
        model = search_space.create_model(sequence=seq, model_input_shape=(128, 128, 3))
        model.save(f"{path}/{uid}.h5")


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
    # sequence = [6, 31, 15, 48, 25, 7, 21, 28, 13, 12, 25, 35, 35, 43, 24, 29]
    if len(sequence) < max_no_of_layers - 1:
        for _ in range((max_no_of_layers - 1) - len(sequence)):
            sequence.append(0)
    acc, lat = trainer.performance_estimate(sequence)
    reward = trainer.multi_objective_reward(acc, lat)
    print("accuracy:", acc, "latency:", lat, "reward:", reward)


def test_full_train():
    search_space = SearchSpaceMn(model_output_shape=2)
    tokens = search_space.generate_token()
    trainer = Trainer(tokens)
    sequence = [41, 8, 9, 43, 25, 34, 33, 9, 43, 35, 48, 49]
    translated = search_space.translate_sequence(sequence)
    architectures = search_space.create_models(samples=[sequence], model_input_shape=(128, 128, 3))
    trained_acc = trainer.train_models(architectures, train_mode="full")[0]
    print(translated)
    print(trained_acc)
