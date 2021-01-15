import numpy as np
import keras

from search_space import SearchSpace


def test_search_space():
    search_space = SearchSpace(target_classes=2)
    vocab = search_space.mapping
    vocab_decoded = search_space.decode_sequence(vocab)
    vocab_encoded = search_space.encode_sequence(vocab_decoded)

    sample_sequence = [8, 8, 30]
    input_shape = np.shape(np.zeros(13))
    model = search_space.create_architecture(sample_sequence, input_shape)
    keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
    assert len(vocab) == 30
    assert len(vocab_decoded) == 30
    assert len(vocab_encoded) == 30
