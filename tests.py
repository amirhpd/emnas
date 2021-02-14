import keras
import numpy as np
from controller import Controller
from search_space import SearchSpace
from trainer import Trainer


def test_search_space():
    search_space = SearchSpace(model_output_shape=1)
    token = search_space.generate_token()

    dense_tokens = [x for x, y in token.items() if "Dense" in y]  # dense layers start from 865
    sample_sequence = [12, 520, 870, 890]
    translated_sequence = search_space.translate_sequence(sample_sequence)
    assert len(translated_sequence) == 4

    model = search_space.create_model(sequence=sample_sequence, model_input_shape=(128, 128, 3))
    keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
    print(model.summary())
    assert len(token) == 890


def test_controller():
    controller = Controller()
    search_space = SearchSpace(model_output_shape=1)
    tokens = search_space.generate_token()

    model = controller.controller_rnn()
    dummy_rnn_input = np.array([[[0, 8, 0, 4]]])  # shape:(1, 1, 4)
    prd = model.predict(dummy_rnn_input)
    assert prd.shape == (1, 1, 890)

    samples = controller.generate_sequence(tokens=tokens)
    architecture = search_space.create_model(sequence=samples[0], model_input_shape=(128, 128, 3))
    print(architecture.summary())
    assert len(samples) == 10

    controller.train_controller_rnn(0)


def test_trainer():
    search_space = SearchSpace(model_output_shape=2)
    tokens = search_space.generate_token()
    controller = Controller(tokens=tokens)
    trainer = Trainer()

    samples = controller.generate_sequence()
    architectures = search_space.create_models(samples=samples, model_input_shape=(128, 128, 3))
    epoch_performance = trainer.train_models(samples=samples, architectures=architectures)
    assert len(epoch_performance) == 0


def test_rnn_trainer():
    search_space = SearchSpace(model_output_shape=2)
    tokens = search_space.generate_token()
    controller = Controller(tokens=tokens)
    samples = controller.generate_sequence()
    manual_epoch_performance = [[[91, 572], 0.6736111044883728],
                                [[466, 262, 372, 85, 52, 572], 0.4930555522441864],
                                [[360, 153, 307, 390, 473, 572], 0.5069444179534912]]

    controller.train_controller_rnn(epoch_performance=manual_epoch_performance)


def test_rnn_performance():
    manipulated_epoch_performance = []
    for i in range(100):
        seq = np.random.randint(low=200, high=571, size=9).tolist() + [572]
        acc = np.random.uniform(0.71, 0.999)
        sample = [seq, acc]
        manipulated_epoch_performance.append(sample)
    for j in range(1000):
        seq = np.random.randint(low=1, high=571, size=9).tolist() + [572]
        acc = np.random.uniform(0.510, 0.999)
        sample = [seq, acc]
        manipulated_epoch_performance.append(sample)

    search_space = SearchSpace(model_output_shape=2)
    tokens = search_space.generate_token()
    controller = Controller(tokens=tokens)

    samples_before_train = controller.generate_sequence()
    controller.train_controller_rnn(epoch_performance=manipulated_epoch_performance)
    samples_after_train = controller.generate_sequence()
    print(samples_before_train)
    print(samples_after_train)

