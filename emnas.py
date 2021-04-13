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
hardware = config.trainer["hardware"]


def plot_image(all_lstm_loss, all_avg_acc, all_avg_lat, all_result):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, gridspec_kw={'hspace': 0.6, 'top': 0.95, 'bottom': 0.05,
                                                                  'right': 0.95, 'left': 0.05,
                                                                  'height_ratios': [1, 1, 1.5, 1, 1]},
                                                  figsize=(20, 24))

    ax1.plot(np.arange(0, len(all_avg_acc)), all_avg_acc)
    ax1.set_title("Average epoch accuracy")
    ax1.set_xlabel("NAS Epochs")
    ax1.grid()
    ax2.plot(np.arange(0, len(all_avg_lat)), all_avg_lat)
    ax2.set_title("Average epoch latency [ms]")
    ax2.set_xlabel("NAS Epochs")
    ax2.grid()
    ax3.plot(np.arange(0, len(all_lstm_loss)), all_lstm_loss)
    ax3.set_title("LSTM loss")
    ax3.set_xlabel("NAS Epochs")
    ax3.grid()
    all_acc = [i[0] for i in list(all_result.values())]
    ax4.plot(np.arange(0, len(all_acc)), all_acc)
    ax4.axhline(y=np.average(all_acc), c="g")
    ax4.set_title(f"All Accuracies ({search_mode})")
    ax4.set_xlabel("Sequences")
    ax4.set_ylim([0, 1])
    ax4.grid()
    all_lat = [i[1] for i in list(all_result.values())]
    ax5.plot(np.arange(0, len(all_lat)), all_lat)
    ax5.axhline(y=np.average(all_lat), c="g")
    ax5.set_title(f"All Latencies [ms] ({hardware})")
    ax5.set_xlabel("Sequences")
    ax5.grid()
    # plt.show()
    return plt


def save_logs(all_lstm_loss, all_avg_acc, all_avg_lat, all_result, final_result, image):
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    path = f"/home/amirhossein/Codes/NAS/emnas/logs/{time_str}_{search_mode}"

    if save_log:
        os.mkdir(path)
        with open(path+"/lstm_loss.txt", "w") as output:
            output.write(str(all_lstm_loss))
        with open(path+"/avg_acc.txt", "w") as output:
            output.write(str(all_avg_acc))
        with open(path+"/avg_lat.txt", "w") as output:
            output.write(str(all_avg_lat))
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
    history_avg_lat = []
    history_result = {}
    for nas_epoch in range(no_of_nas_epochs):
        print(f"NAS epoch {nas_epoch+1}/{no_of_nas_epochs}:")
        samples = controller.generate_sequence()

        print(f"Training {len(samples)} architectures:")
        architectures = search_space.create_models(samples=samples, model_input_shape=model_input_shape)
        epoch_performance = trainer.train_models(samples=samples, architectures=architectures)
        history_result.update(epoch_performance)
        avg_acc = round(np.average([i[0] for i in list(epoch_performance.values())]), 3)
        avg_lat = round(np.average([i[1] for i in list(epoch_performance.values())]), 3)
        history_avg_acc.append(avg_acc)
        history_avg_lat.append(avg_lat)
        print("Epoch average accuracy:", avg_acc, "average latency:", avg_lat)

        print("Training controller:")
        lstm_loss = controller.train_controller_rnn(epoch_performance=epoch_performance)
        history_lstm_loss.append(lstm_loss)
        print("Controller average loss:", lstm_loss)
        print("---------------------------------------------")
        print("---------------------------------------------")

    t2 = round(time.time() - t1, 2)
    best = max(history_result, key=history_result.get)  # based on only accuracy
    best_translated = search_space.translate_sequence(best)
    acc_average = round(np.average([i[0] for i in list(history_result.values())]), 3)
    lat_average = round(np.average([i[1] for i in list(history_result.values())]), 3)
    final_info = [best_translated, history_result[best], acc_average, lat_average, t2]

    img = plot_image(history_lstm_loss, history_avg_acc, history_avg_lat, history_result)
    save_logs(history_lstm_loss, history_avg_acc, history_avg_lat, history_result, final_info, img)

    print("Best architecture:")
    print(final_info[0])
    print("With accuracy:", final_info[1][0], "and latency:", final_info[1][1], "ms")
    print("Total average accuracy:", acc_average)
    print("Total average latency:", lat_average, "ms")
    print("NAS done in", t2, "sec")


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
            if [i[0] for i in list(epoch_performance.values())][0] >= naive_threshold:
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
            if [i[0] for i in list(epoch_performance.values())][0] >= naive_threshold:
                result = epoch_performance
                break

    t2 = round(time.time() - t1, 2)
    best_translated = search_space.translate_sequence(list(result.keys())[0])
    if result:
        print("Found architecture:")
        print(best_translated)
        print(f"With accuracy: {round([i[0] for i in list(result.values())][0], 2)} and latency "
              f"{round([i[1] for i in list(result.values())][0], 2)} ms "
              f"after checking {cnt_valid} sequences and skipping {cnt_skip} sequences.")
        print("NAS done in", t2, "sec")
    else:
        print(f"No architecture with accuracy >= {naive_threshold} found.")
        print("NAS done in", t2, "sec")

    acc_average = np.average([i[0] for i in list(result.values())][0])
    lat_average = np.average([i[0] for i in list(result.values())][0])
    final_info = [best_translated, list(result.values())[0], acc_average, lat_average, t2, cnt_valid, cnt_skip]

    img = plot_image(all_lstm_loss=[0], all_avg_acc=[0], all_avg_lat=[0], all_result=history_result)
    save_logs(all_lstm_loss=[0], all_avg_acc=[0], all_avg_lat=[0], all_result=history_result,
              final_result=final_info, image=img)


if __name__ == '__main__':
    save_log = True
    print("MODE:", search_mode, "HARDWARE:", hardware)
    if search_mode == "rnn":
        main_rnn()
    else:
        main_naive()
