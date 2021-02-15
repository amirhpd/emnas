from controller import Controller
from search_space import SearchSpace
from trainer import Trainer
import config
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


no_of_nas_epochs = config.emnas["no_of_nas_epochs"]
model_output_shape = config.emnas["model_output_shape"]
model_input_shape = config.emnas["model_input_shape"]


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
    ax3.set_title("All Accuracies")
    ax3.set_xlabel("Sequences")
    ax3.set_ylim([0, 1])
    ax3.grid()
    # plt.show()
    return plt


def save_logs(all_lstm_loss, all_avg_acc, all_result, best_model, image):
    name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    path = "/home/amirhossein/Codes/NAS/emnas/logs/" + name
    os.mkdir(path)

    with open(path+"/lstm_loss.txt", "w") as output:
        output.write(str(all_lstm_loss))
    with open(path+"/avg_acc.txt", "w") as output:
        output.write(str(all_avg_acc))
    with open(path+"/results.txt", "w") as output:
        output.write(str(all_result))
    with open(path+"/best_model.txt", "w") as output:
        output.write(str(best_model))
    with open(path+"/config.txt", "w") as output:
        output.write(str(config.emnas))
        output.write("\n")
        output.write(str(config.controller))
        output.write("\n")
        output.write(str(config.trainer))
        output.write("\n")
        output.write(str(config.search_space))

    image.savefig(path+"/graphs.png")
    image.show()


if __name__ == '__main__':
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

    best = max(history_result, key=history_result.get)
    best_translated = search_space.translate_sequence(best)
    best_info = [best_translated, history_result[best]]
    print("Best architecture:")
    print(best_info[0])
    print("With accuracy:", best_info[1])
    print(f"NAS done in {round(time.time() - t1, 2)}s")
    img = plot_image(history_lstm_loss, history_avg_acc, history_result)
    save_logs(history_lstm_loss, history_avg_acc, history_result, best_info, img)


