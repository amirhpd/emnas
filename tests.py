import keras
from search_space import SearchSpace


def test_search_space():
    search_space = SearchSpace(model_output_shape=1)
    tokens = search_space.generate_token()

    sample_sequence = ["c8", "c52", "d14", "out"]
    translated_sequence = search_space.translate_sequence(sample_sequence)

    model = search_space.create_model(sample_sequence, (128, 128, 3))
    keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
    print(model.summary())
    assert len(tokens) == 890
