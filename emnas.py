from controller import Controller
from search_space import SearchSpace
from trainer import Trainer
import config
import numpy as np
import time
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


save_log = True
no_of_nas_epochs = config.emnas["no_of_nas_epochs"]
model_output_shape = config.emnas["model_output_shape"]
model_input_shape = config.emnas["model_input_shape"]
search_mode = config.emnas["search_mode"]
naive_threshold = config.emnas["naive_threshold"]
naive_timeout = config.emnas["naive_timeout"]


def plot_image(all_lstm_loss, all_avg_acc, all_result):
    fig, (ax1, ax2, ax3) = plt.subplots(3, gridspec_kw={'hspace': 0.4, 'top': 0.95, 'bottom': 0.05,
                                                        'right': 0.95, 'left': 0.05, 'height_ratios': [1, 1, 1.5]},
                                        figsize=(20, 10))

    ax1.plot(np.arange(0, len(all_avg_acc)), all_avg_acc)
    ax1.set_title("Average epoch accuracy")
    ax1.set_xlabel("NAS Epochs")
    ax1.grid()
    ax2.plot(np.arange(0, len(all_lstm_loss)), all_lstm_loss)
    ax2.set_title("LSTM loss")
    ax2.set_xlabel("NAS Epochs")
    ax2.grid()
    all_acc = list(all_result.values())
    ax3.plot(np.arange(0, len(all_acc)), all_acc)
    ax3.axhline(y=np.average(all_acc), c="g")
    ax3.set_title(f"All Accuracies ({search_mode})")
    ax3.set_xlabel("Sequences")
    ax3.set_ylim([0, 1])
    ax3.grid()
    # plt.show()
    return plt


def save_logs(all_lstm_loss, all_avg_acc, all_result, final_result, image):
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    path = f"/home/amirhossein/Codes/NAS/emnas/logs/{time_str}_{search_mode}"

    if save_log:
        os.mkdir(path)
        with open(path+"/lstm_loss.txt", "w") as output:
            output.write(str(all_lstm_loss))
        with open(path+"/avg_acc.txt", "w") as output:
            output.write(str(all_avg_acc))
        with open(path+"/results.txt", "w") as output:
            output.write(str(all_result))
        with open(path+"/final_info.txt", "w") as output:
            output.write(str(final_result))
        with open(path+"/config.txt", "w") as output:
            output.write(str(config.emnas))
            output.write("\n")
            output.write(str(config.controller))
            output.write("\n")
            output.write(str(config.trainer))
            output.write("\n")
            output.write(str(config.search_space))
        shutil.copyfile("search_space.json", path+"/search_space.json")

        image.savefig(path+"/graphs.png")
    image.show()


def main_rnn():
    t1 = time.time()
    search_space = SearchSpace(model_output_shape=model_output_shape)
    tokens = search_space.generate_token()
    controller = Controller(tokens=tokens)
    trainer = Trainer()

    history_lstm_loss = []
    history_avg_acc = []
    history_result = {}
    for nas_epoch in range(no_of_nas_epochs):
        print(f"NAS epoch {nas_epoch+1}/{no_of_nas_epochs}:")
        samples = controller.generate_sequence()
        architectures = search_space.create_models(samples=samples, model_input_shape=model_input_shape)

        print(f"Training {len(samples)} architectures:")
        epoch_performance = trainer.train_models(samples=samples, architectures=architectures)
        history_result.update(epoch_performance)
        avg_acc = np.average(list(epoch_performance.values()))
        history_avg_acc.append(avg_acc)
        print("Epoch average accuracy:", avg_acc)

        print("Training controller:")
        lstm_loss = controller.train_controller_rnn(epoch_performance=epoch_performance)
        history_lstm_loss.append(lstm_loss)
        print("Controller average loss:", lstm_loss)
        print("---------------------------------------------")
        print("---------------------------------------------")

    t2 = round(time.time() - t1, 2)
    best = max(history_result, key=history_result.get)
    best_translated = search_space.translate_sequence(best)
    acc_average = np.average(list(history_result.values()))
    final_info = [best_translated, history_result[best], acc_average, t2]

    img = plot_image(all_lstm_loss=history_lstm_loss, all_avg_acc=history_avg_acc, all_result=history_result)
    save_logs(history_lstm_loss, history_avg_acc, history_result, final_info, img)

    print("Best architecture:")
    print(final_info[0])
    print("With accuracy:", final_info[1])
    print("Total average accuracy:", acc_average)
    print(f"NAS done in {t2}s")


def main_naive():
    t1 = time.time()
    search_space = SearchSpace(model_output_shape=model_output_shape)
    tokens = search_space.generate_token()
    controller = Controller(tokens=tokens)
    trainer = Trainer()
    result = None
    history_result = {}
    cnt_valid = 1
    cnt_skip = 1

    if search_mode == "bruteforce":
        space = controller.generate_sequence_naive(mode="b")
        for sequence in space:
            sequence = sequence + (list(tokens.keys())[-1],)  # add last layer
            if not controller.check_sequence(sequence):
                cnt_skip += 1
                continue
            cnt_valid += 1
            architecture = search_space.create_model(sequence=sequence, model_input_shape=model_input_shape)
            epoch_performance = trainer.train_models(samples=[sequence], architectures=[architecture])
            history_result.update(epoch_performance)
            if list(epoch_performance.values())[0] >= naive_threshold:
                result = epoch_performance
                break

    if search_mode == "random":
        watchdog = 0
        while watchdog < naive_timeout:
            watchdog += 1
            sequence = controller.generate_sequence_naive(mode="r") + [list(tokens.keys())[-1]]  # add last layer
            if (tuple(sequence) in list(history_result.keys())) or (not controller.check_sequence(sequence)):
                cnt_skip += 1
                continue
            cnt_valid += 1
            architecture = search_space.create_model(sequence=sequence, model_input_shape=model_input_shape)
            epoch_performance = trainer.train_models(samples=[sequence], architectures=[architecture])
            history_result.update(epoch_performance)
            if list(epoch_performance.values())[0] >= naive_threshold:
                result = epoch_performance
                break

    t2 = round(time.time() - t1, 2)
    best_translated = search_space.translate_sequence(list(result.keys())[0])
    if result:
        print("Found architecture:")
        print(best_translated)
        print(f"With accuracy: {list(result.values())[0]} after checking {cnt_valid} sequences and skipping {cnt_skip} sequences.")
        print(f"DONE (t:{t2})")
    else:
        print(f"No architecture with accuracy >= {naive_threshold} found.")
        print(f"DONE (t:{t2})")

    acc_average = np.average(list(history_result.values()))
    final_info = [best_translated, list(result.values())[0], acc_average, t2, cnt_valid, cnt_skip]

    img = plot_image(all_lstm_loss=[0], all_avg_acc=[0], all_result=history_result)
    save_logs(all_lstm_loss=[0], all_avg_acc=[0], all_result=history_result, final_result=final_info, image=img)


def _generate_init(history_result):
    print("generating init ..")
    search_space = SearchSpace(model_output_shape=model_output_shape)
    tokens = search_space.generate_token()
    controller = Controller(tokens=tokens)
    trainer = Trainer()

    while True:
        init = controller.generate_sequence_naive(mode="r") + [list(tokens.keys())[-1]]  # add last layer
        if (tuple(init) in list(history_result.keys())) or (not controller.check_sequence(init)):
            continue
        init_architecture = search_space.create_model(sequence=init, model_input_shape=model_input_shape)
        init_epoch_performance = trainer.train_models(samples=[init], architectures=[init_architecture])
        if list(init_epoch_performance.values())[0] >= 0.7:
            history_result.update(init_epoch_performance)
            print("init:")
            print(init_epoch_performance)
            break

    return init, history_result, init_epoch_performance


def main_hill_climbing():
    t1 = time.time()
    search_space = SearchSpace(model_output_shape=model_output_shape)
    tokens = search_space.generate_token()
    controller = Controller(tokens=tokens)
    trainer = Trainer()
    history_result = {}
    cnt_valid = 1
    cnt_skip = 1

    init, history_result, init_epoch_performance = _generate_init(history_result)

    for i in range(10):
        successors = controller.generate_successors(init)
        winning_successor = 0
        for successor in successors:
            if (tuple(successor) in list(history_result.keys())) or (not controller.check_sequence(successor)):
                cnt_skip += 1
                continue
            cnt_valid += 1
            architecture = search_space.create_model(sequence=successor, model_input_shape=model_input_shape)
            epoch_performance = trainer.train_models(samples=[successor], architectures=[architecture])
            history_result.update(epoch_performance)
            if list(epoch_performance.values())[0] >= list(init_epoch_performance.values())[0]:
                print("winning successor:", epoch_performance)
                init = successor
                init_epoch_performance = epoch_performance
                winning_successor = 1
                break
        if winning_successor == 0:
            print("no winning successor")
            init, history_result, init_epoch_performance = _generate_init(history_result)

    print(history_result)


if __name__ == '__main__':
    save_log = True
    print("MODE:", search_mode)
    if search_mode == "rnn":
        main_rnn()
    elif search_mode == "hill_climbing":
        main_hill_climbing()
    else:
        main_naive()
