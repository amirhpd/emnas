import keras
import numpy as np
from controller import Controller
from emnas import plot_image, save_logs
from search_space import SearchSpace
from trainer import Trainer
from camera_drive import SipeedCamera


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


def test_trainer():
    search_space = SearchSpace(model_output_shape=2)
    tokens = search_space.generate_token()
    controller = Controller(tokens=tokens)
    trainer = Trainer()

    samples = controller.generate_sequence()
    architectures = search_space.create_models(samples=samples, model_input_shape=(128, 128, 3))
    epoch_performance = trainer.train_models(samples=samples, architectures=architectures)
    assert len(epoch_performance) == 0


def test_controller_rnn_trainer():
    search_space = SearchSpace(model_output_shape=2)
    tokens = search_space.generate_token()
    controller = Controller(tokens=tokens)
    samples = controller.generate_sequence()
    manual_epoch_performance = {
        (466, 262, 372, 85, 52, 572): 0.4930555522441864,
        (360, 153, 307, 390, 473, 572): 0.5069444179534912,
        (315, 476, 563, 565, 560, 572): 0.8032407164573669,
        (318, 291, 20, 326, 161, 572): 0.8055555820465088
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

    sequences_random = controller.generate_sequence_naive(mode="r")
    print(sequences_random)


def test_plot_image_logs():
    history_lstm_loss = [2.570976454421725e-05, -4.0828806140780216e-06, 2.319243887882294e-05, 1.2543778396789662e-05,
                         -1.1891249442612662e-05, -1.734991667262875e-05, 1.366215491316325e-07, 1.676019123806327e-05,
                         1.005385006465076e-06, -9.377887454320444e-07]
    history_avg_acc = [0.724, 0.651, 0.686, 0.688, 0.641, 0.723, 0.667, 0.667, 0.761, 0.669]
    history_result = {
        (468, 38, 248, 544, 558, 572): 0.7662037014961243,
        (467, 17, 20, 292, 483, 572): 0.5069444179534912,
        (12, 99, 378, 420, 246, 572): 0.7916666865348816,
        (151, 143, 482, 282, 131, 572): 0.7569444179534912,
        (259, 304, 97, 491, 358, 572): 0.5069444179534912,
        (355, 118, 479, 307, 373, 572): 0.4930555522441864
    }
    best_model = [
        [('DepthwiseConv2D', 16, (1, 1), (1, 1), 'same', 'tanh'),
         ('Conv2D', 40, (3, 3), (2, 2), 'valid', 'tanh'),
         ('Conv2D', 32, (1, 1), (2, 2), 'valid', 'relu'),
         ('Conv2D', 40, (1, 1), (1, 1), 'same', 'tanh'),
         ('DepthwiseConv2D', 24, (2, 2), (1, 1), 'same', 'tanh'), (2, 'softmax')],
        0.8402777910232544]

    #  rnn
    # img = plot_image(history_lstm_loss, history_avg_acc, history_result)
    # save_logs(history_lstm_loss, history_avg_acc, history_result, best_model, img)

    #  naive
    img = plot_image(all_lstm_loss=[0], all_avg_acc=[0], all_result=history_result)
    save_logs(all_lstm_loss=[0], all_avg_acc=[0], all_result=history_result, final_result=best_model, image=img)


def test_sipeed_get_latency():
    sipeed_cam = SipeedCamera()
    latency = sipeed_cam.get_latency(model_file="model_6.kmodel")
    print(latency)
