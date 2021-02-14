from controller import Controller
from search_space import SearchSpace


def test_controller():
    search_space = SearchSpace(model_output_shape=2)
    tokens = search_space.generate_token()
    controller = Controller(tokens=tokens)
    samples = controller.generate_sequence_random_predict()
    print(samples)


def test_check_sequence():
    search_space = SearchSpace(model_output_shape=2)
    tokens = search_space.generate_token()
    controller = Controller(tokens=tokens)
    valid_sequences = [[152, 140, 445, 260, 530, 572],
                       [292, 226, 169, 328, 557, 572],
                       [517, 411, 522, 242, 239, 572]]

    invalid_sequences = [[152, 140, 445, 260, 530, 530],
                         [572, 140, 169, 328, 557, 572],
                         [517, 572, 522, 242, 239, 572],
                         [152, 140, 542, 260, 530, 572]]

    for sequence in valid_sequences:
        result = controller.check_sequence(sequence)
        assert result

    for sequence in invalid_sequences:
        result = controller.check_sequence(sequence)
        assert not result
