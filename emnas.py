import pandas as pd
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
import cv2

matplotlib.use('TkAgg')

no_of_nas_epochs = config.emnas["no_of_nas_epochs"]
model_output_shape = config.emnas["model_output_shape"]
model_input_shape = config.emnas["model_input_shape"]
search_mode = config.emnas["search_mode"]
naive_threshold = config.emnas["naive_threshold"]
naive_timeout = config.emnas["naive_timeout"]
hardware = config.trainer["hardware"]
no_of_episodes = config.emnas["no_of_episodes"]
log_path = config.emnas["log_path"]


def _plot(history, path):
    fig = plt.figure(figsize=(24, 4))
    plt.plot(np.arange(0, len(history["loss"])), history["loss"])
    plt.title("Agent loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(path + "/fig_1.png")

    fig = plt.figure(figsize=(24, 4))
    plt.plot(np.arange(0, len(history["accuracy_per_episode"])), history["accuracy_per_episode"],
             label="Last accuracy of episode")
    plt.plot(np.arange(0, len(history["avg_accuracy_per_episode"])), history["avg_accuracy_per_episode"],
             label="Average accuracy of episode")
    plt.axhline(y=np.average(history["accuracy_per_episode"]), c="g", label="Total average")
    plt.title("Accuracy results per episode")
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend()
    plt.savefig(path + "/fig_2.png")

    fig = plt.figure(figsize=(24, 4))
    bins = int(len(history["accuracy_per_episode"]) / 5)
    plt.hist(history["accuracy_per_episode"], bins=bins, edgecolor="k")
    plt.title("Accuracy distribution per episode")
    plt.xlabel("Accuracy")
    plt.ylabel("Count")
    plt.grid()
    plt.savefig(path + "/fig_3.png")

    min_acc = [i[0] for i in history["min_max"]]
    fig = plt.figure(figsize=(24, 4))
    plt.plot(np.arange(0, len(min_acc)), min_acc)
    plt.axhline(y=np.average(min_acc), c="g")
    plt.title("Minimum accuracy of episode")
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig(path + "/fig_4.png")

    max_acc = [i[1] for i in history["min_max"]]
    fig = plt.figure(figsize=(24, 4))
    plt.plot(np.arange(0, len(max_acc)), max_acc)
    plt.axhline(y=np.average(max_acc), c="g")
    plt.title("Maximum accuracy of episode")
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig(path + "/fig_5.png")

    fig = plt.figure(figsize=(24, 4))
    plt.plot(np.arange(0, len(history["best_so_far"])), history["best_so_far"])
    plt.title("Best accuracy so far")
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig(path + "/fig_6.png")

    fig = plt.figure(figsize=(24, 4))
    plt.plot(np.arange(0, len(history["accuracy_per_play"])), history["accuracy_per_play"])
    plt.axhline(y=np.average(history["accuracy_per_play"]), c="g")
    plt.title("All accuracies per play")
    plt.xlabel("Play")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig(path + "/fig_7.png")

    fig = plt.figure(figsize=(24, 4))
    bins = int(len(history["accuracy_per_play"]) / 5)
    plt.hist(history["accuracy_per_play"], bins=bins, edgecolor="k")
    plt.title("Accuracy distribution per play")
    plt.xlabel("Accuracy")
    plt.ylabel("Count")
    plt.grid()
    plt.savefig(path + "/fig_8.png")

    fig = plt.figure(figsize=(24, 4))
    plt.plot(np.arange(0, len(history["play_counts"])), history["play_counts"])
    plt.title("No. of plays in one episode")
    plt.xlabel("Episode")
    plt.ylabel("Count")
    plt.grid()
    plt.savefig(path + "/fig_9.png")

    img_list = [cv2.imread(path + f"/fig_{i}.png") for i in range(1, 10)]
    img = cv2.vconcat(img_list)
    cv2.imwrite(path + "/fig.png", img)
    [os.remove(path + f"/fig_{i}.png") for i in range(1, 10)]


def save_logs(history, current_logs, best_result):
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    path = f"{log_path}/{time_str}_{search_mode}"
    os.mkdir(path)
    _plot(history, path)

    history_play = dict((k, history[k]) for k in ("accuracy_per_play", "sequence_per_play"))
    history.pop("accuracy_per_play", None)
    history.pop("sequence_per_play", None)
    df_play = pd.DataFrame(history_play)
    df_episode = pd.DataFrame(history)
    df_play.to_csv(path + "/play_data.csv", index=False)
    df_episode.to_csv(path + "/episode_data.csv", index=False)

    shutil.copyfile("search_space.json", path + "/search_space.json")
    shutil.copyfile("config.py", path + "/config.txt")

    df_current_logs = pd.DataFrame(current_logs)
    df_current_logs.to_csv(path + "/current_logs.csv", index=False)

    with open(path + "/best_result.txt", "w") as output:
        output.write(best_result)


def main_ff():
    t1 = time.time()
    search_space = SearchSpace(model_output_shape=model_output_shape)
    tokens = search_space.generate_token()
    controller = Controller(tokens=tokens)
    trainer = Trainer(tokens)

    history = {
        "accuracy_per_play": [],
        "sequence_per_play": [],
        "sequence_per_episode": [],
        "accuracy_per_episode": [],
        "avg_accuracy_per_episode": [],
        "loss": [],
        "play_counts": [],
        "min_max": [],
        "best_so_far": [],
    }
    current_logs = []

    sequence = np.random.randint(1, controller.len_search_space, controller.max_no_of_layers - 1,
                                 dtype="int32")[np.newaxis]
    for episode in range(no_of_episodes):
        done = False
        play_counter = 0
        episode_acc = []
        while not done:
            play_counter += 1

            actions, prob = controller.get_action(sequence)
            reward = trainer.performance_estimate(actions)
            done = False
            if reward < controller.min_reward and play_counter >= controller.min_plays:
                done = True
            if play_counter >= controller.max_plays:
                done = True

            controller.remember(sequence, actions, prob, reward)
            sequence = np.array(actions)[np.newaxis]
            episode_acc.append(reward)
            history["accuracy_per_play"].append(reward)
            history["sequence_per_play"].append(sequence)

            if len(episode_acc) >= controller.min_plays and np.var(episode_acc) < 1e-2:
                done = True

            if done:
                history["min_max"].append([controller.min_reward, max(episode_acc)])
                history["sequence_per_episode"].append(sequence)
                history["accuracy_per_episode"].append(reward)
                history["play_counts"].append(play_counter)
                history["best_so_far"].append(max(history["accuracy_per_play"]))
                history["avg_accuracy_per_episode"].append(np.average(episode_acc))
                sequence = history["sequence_per_play"][
                    history["accuracy_per_play"].index(max(history["accuracy_per_play"]))]

                loss_value = controller.update_policy()
                history["loss"].append(loss_value)

        current_log = {
            "Episode": episode + 1,
            "Plays:": play_counter,
            "Max accuracy": round(max(episode_acc), 3),
            "reward_min": round(controller.min_reward, 3),
            "Best accuracy:": max(history["accuracy_per_play"]),
            "Loss": loss_value,
        }
        for k, v in current_log.items():
            print(k, ":", end=" ")
            print(v, "\t", end=" ")
        print()
        current_logs.append(current_log)

    best_sequence = history["sequence_per_play"][
        history["accuracy_per_play"].index(max(history["accuracy_per_play"]))]
    best_result = f"Best sequence: {best_sequence}" \
                  f" \t with accuracy: {round(trainer.performance_estimate(best_sequence[0]), 3)}" \
                  f" \t Explored: {len(history['sequence_per_play'])} \t In {int(time.time() - t1)} sec. \n" \
                  f"Total average accuracy: {np.average(history['accuracy_per_play'])}"
    print(best_result)

    save_logs(history, current_logs, best_result)
    print("DONE")


# def main_naive():
#     t1 = time.time()
#     search_space = SearchSpace(model_output_shape=model_output_shape)
#     tokens = search_space.generate_token()
#     controller = Controller(tokens=tokens)
#     trainer = Trainer(tokens)
#     result = None
#     history_result = {}
#     cnt_valid = 1
#     cnt_skip = 1
#
#     if search_mode == "bruteforce":
#         space = controller.generate_sequence_naive(mode="b")
#         for sequence in space:
#             sequence = sequence + (list(tokens.keys())[-1],)  # add last layer
#             if not controller.check_sequence(sequence):
#                 cnt_skip += 1
#                 continue
#             cnt_valid += 1
#             architecture = search_space.create_model(sequence=sequence, model_input_shape=model_input_shape)
#             epoch_performance = trainer.train_models(samples=[sequence], architectures=[architecture])
#             history_result.update(epoch_performance)
#             if [i[0] for i in list(epoch_performance.values())][0] >= naive_threshold:
#                 result = epoch_performance
#                 break
#
#     if search_mode == "random":
#         watchdog = 0
#         while watchdog < naive_timeout:
#             watchdog += 1
#             sequence = controller.generate_sequence_naive(mode="r") + [list(tokens.keys())[-1]]  # add last layer
#             if (tuple(sequence) in list(history_result.keys())) or (not controller.check_sequence(sequence)):
#                 cnt_skip += 1
#                 continue
#             cnt_valid += 1
#             architecture = search_space.create_models(samples=[sequence], model_input_shape=model_input_shape)
#             epoch_performance = trainer.train_models(samples=[sequence], architectures=architecture)
#             history_result.update(epoch_performance)
#             if [i[0] for i in list(epoch_performance.values())][0] >= naive_threshold:
#                 result = epoch_performance
#                 break
#
#     t2 = round(time.time() - t1, 2)
#     best_translated = search_space.translate_sequence(list(result.keys())[0])
#     if result:
#         print("Found architecture:")
#         print(best_translated)
#         print(f"With accuracy: {round([i[0] for i in list(result.values())][0], 2)} and latency "
#               f"{round([i[1] for i in list(result.values())][0], 2)} ms "
#               f"after checking {cnt_valid} sequences and skipping {cnt_skip} sequences.")
#         print("NAS done in", t2, "sec")
#     else:
#         print(f"No architecture with accuracy >= {naive_threshold} found.")
#         print("NAS done in", t2, "sec")
#
#     acc_average = np.average([i[0] for i in list(result.values())][0])
#     lat_average = np.average([i[0] for i in list(result.values())][0])
#     final_info = [best_translated, list(result.values())[0], acc_average, lat_average, t2, cnt_valid, cnt_skip]
#
#     img = plot_image(all_lstm_loss=[0], all_avg_acc=[0], all_avg_lat=[0], all_result=history_result)
#     save_logs(all_lstm_loss=[0], all_avg_acc=[0], all_avg_lat=[0], all_result=history_result,
#               final_result=final_info, image=img)


if __name__ == '__main__':
    print("MODE:", search_mode, "HARDWARE:", hardware)
    if search_mode == "ff":
        main_ff()
    # else:
    #     main_naive()
