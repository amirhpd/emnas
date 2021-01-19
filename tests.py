import keras
import numpy as np
from controller import Controller
from search_space import SearchSpace


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
    controller = Controller(search_space_length=890)
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
