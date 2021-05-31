import pandas as pd
from controller import Controller
from search_space import SearchSpace
from search_space_mn import SearchSpaceMn
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

model_output_shape = config.emnas["model_output_shape"]
model_input_shape = config.emnas["model_input_shape"]
search_mode = config.emnas["search_mode"]
naive_threshold = config.emnas["naive_threshold"]
naive_timeout = config.emnas["naive_timeout"]
no_of_episodes = config.emnas["no_of_episodes"]
log_path = config.emnas["log_path"]
max_no_of_layers = config.controller["max_no_of_layers"]
dynamic_min_reward = config.controller["dynamic_min_reward"]
variance_threshold = config.controller["variance_threshold"]
valid_actions = config.controller["valid_actions"]

if config.search_space["mode"] == "MobileNets":
    search_space = SearchSpaceMn(model_output_shape=model_output_shape)
else:
    search_space = SearchSpace(model_output_shape=model_output_shape)
tokens = search_space.generate_token()
controller = Controller(tokens=tokens)
trainer = Trainer(tokens)


def _plot(history, path):
    fig = plt.figure(figsize=(24, 4))
    plt.plot(np.arange(0, len(history["loss"])), history["loss"])
    plt.title("Agent loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(path + "/fig_1.png")

    fig = plt.figure(figsize=(24, 4))
    plt.plot(np.arange(0, len(history["reward_per_episode"])), history["reward_per_episode"],
             label="Last reward of episode")
    plt.plot(np.arange(0, len(history["avg_reward_per_episode"])), history["avg_reward_per_episode"],
             label="Average reward of episode")
    plt.axhline(y=np.average(history["reward_per_episode"]), c="g", label="Total reward")
    plt.title("Reward per episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.legend()
    plt.savefig(path + "/fig_2.png")

    fig = plt.figure(figsize=(24, 4))
    bins = int(len(history["reward_per_episode"]) / 5)
    bins = bins if bins > 1 else 1
    plt.hist(history["reward_per_episode"], bins=bins, edgecolor="k")
    plt.title("Reward distribution per episode")
    plt.xlabel("Reward")
    plt.ylabel("Count")
    plt.grid()
    plt.savefig(path + "/fig_3.png")

    fig = plt.figure(figsize=(24, 4))
    min_acc = [i[0] for i in history["min_max"]]
    plt.plot(np.arange(0, len(min_acc)), min_acc)
    plt.axhline(y=np.average(min_acc), c="g")
    plt.title("Minimum reward of episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.savefig(path + "/fig_4.png")

    fig = plt.figure(figsize=(24, 4))
    max_acc = [i[1] for i in history["min_max"]]
    plt.plot(np.arange(0, len(max_acc)), max_acc)
    plt.axhline(y=np.average(max_acc), c="g")
    plt.title("Maximum reward of episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.savefig(path + "/fig_5.png")

    fig = plt.figure(figsize=(24, 4))
    best_acc = [i[0] for i in history["best_so_far"]]
    plt.plot(np.arange(0, len(best_acc)), best_acc)
    plt.title("Best accuracy so far")
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig(path + "/fig_6.png")

    fig = plt.figure(figsize=(24, 4))
    best_lat = [i[1] for i in history["best_so_far"]]
    plt.plot(np.arange(0, len(best_lat)), best_lat)
    plt.title("Best latency so far")
    plt.xlabel("Episode")
    plt.ylabel("Latency [ms]")
    plt.grid()
    plt.savefig(path + "/fig_7.png")

    fig = plt.figure(figsize=(24, 4))
    best_rew = [i[2] for i in history["best_so_far"]]
    plt.plot(np.arange(0, len(best_rew)), best_rew)
    plt.title("Best reward so far")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.savefig(path + "/fig_8.png")

    fig = plt.figure(figsize=(24, 4))
    plt.plot(np.arange(0, len(history["reward_per_play"])), history["reward_per_play"])
    plt.axhline(y=np.average(history["reward_per_play"]), c="g")
    plt.title("All rewards per play")
    plt.xlabel("Play")
    plt.ylabel("Reward")
    plt.grid()
    plt.savefig(path + "/fig_9.png")

    fig = plt.figure(figsize=(24, 4))
    bins = int(len(history["reward_per_play"]) / 5)
    bins = bins if bins > 1 else 1
    plt.hist(history["reward_per_play"], bins=bins, edgecolor="k")
    plt.title("Reward distribution per play")
    plt.xlabel("Reward")
    plt.ylabel("Count")
    plt.grid()
    plt.savefig(path + "/fig_10.png")

    fig = plt.figure(figsize=(24, 4))
    plt.plot(np.arange(0, len(history["play_counts"])), history["play_counts"])
    plt.title("No. of plays in one episode")
    plt.xlabel("Episode")
    plt.ylabel("Count")
    plt.grid()
    plt.savefig(path + "/fig_11.png")

    img_list = [cv2.imread(path + f"/fig_{i}.png") for i in range(1, 12)]
    img = cv2.vconcat(img_list)
    cv2.imwrite(path + "/fig.png", img)
    [os.remove(path + f"/fig_{i}.png") for i in range(1, 12)]


def save_logs(history, current_logs, best_result):
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    path = f"{log_path}/{time_str}_{search_mode}"
    os.mkdir(path)
    _plot(history, path)

    history_play = dict((k, history[k]) for k in ("accuracy_per_play", "latency_per_play", "reward_per_play",
                                                  "sequence_per_play"))
    history.pop("accuracy_per_play", None)
    history.pop("latency_per_play", None)
    history.pop("reward_per_play", None)
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


def main_rl():
    t1 = time.time()
    history = {
        "accuracy_per_play": [],
        "latency_per_play": [],
        "reward_per_play": [],
        "sequence_per_play": [],
        "sequence_per_episode": [],
        "accuracy_per_episode": [],
        "latency_per_episode": [],
        "reward_per_episode": [],
        "avg_reward_per_episode": [],
        "loss": [],
        "play_counts": [],
        "min_max": [],
        "best_so_far": [],
    }
    current_logs = []
    loss_value = 0
    min_reward = controller.min_reward

    sequence = np.random.randint(1, controller.len_search_space, controller.max_no_of_layers - 1,
                                 dtype="int32")[np.newaxis]
    for episode in range(no_of_episodes):
        done = False
        play_counter = 0
        invalid_counter = 0
        episode_rew = []
        while not done:
            if valid_actions:
                actions, prob, valid = controller.get_valid_action(sequence)
                accuracy, latency = trainer.performance_estimate(actions)
                reward = trainer.multi_objective_reward(accuracy, latency)
                play_counter += 1
                invalid_counter = valid
            else:
                actions, prob, valid = controller.get_all_action(sequence)
                if valid:
                    accuracy, latency = trainer.performance_estimate(actions)
                    reward = trainer.multi_objective_reward(accuracy, latency)
                    play_counter += 1
                else:
                    invalid_counter += 1
                    reward = 0.4

            done = False
            if reward < min_reward and play_counter >= controller.min_plays:
                done = True
            if play_counter >= controller.max_plays:
                done = True

            controller.remember(sequence, actions, prob, reward)
            sequence = np.array(actions)[np.newaxis]
            episode_rew.append(reward)
            history["accuracy_per_play"].append(accuracy)
            history["latency_per_play"].append(latency)
            history["reward_per_play"].append(reward)
            history["sequence_per_play"].append(sequence)

            if play_counter >= controller.min_plays and np.var(episode_rew) < variance_threshold:
                done = True

            if done:
                history["min_max"].append([min_reward, max(episode_rew)])
                history["sequence_per_episode"].append(sequence)
                history["accuracy_per_episode"].append(accuracy)
                history["latency_per_episode"].append(latency)
                history["reward_per_episode"].append(reward)
                history["play_counts"].append(play_counter)
                history["avg_reward_per_episode"].append(np.average(episode_rew))
                sequence = history["sequence_per_play"][
                    history["reward_per_play"].index(max(history["reward_per_play"]))]
                best_reward_index = history["reward_per_play"].index(max(history["reward_per_play"]))
                history["best_so_far"].append((
                    history["accuracy_per_play"][best_reward_index],
                    history["latency_per_play"][best_reward_index],
                    history["reward_per_play"][best_reward_index]
                ))

                if dynamic_min_reward:
                    min_reward = np.average(episode_rew) if np.average(episode_rew) > min_reward else min_reward

                loss_value = controller.update_policy()
                history["loss"].append(loss_value)

        current_log = {
            "Episode": episode + 1,
            "Plays:": play_counter,
            "Max reward": round(max(episode_rew), 3),
            "reward_min": round(min_reward, 3),
            "Best reward:": max(history["reward_per_play"]),
            "Loss": loss_value,
            "Invalids": invalid_counter,
        }
        for k, v in current_log.items():
            print(k, ":", end=" ")
            print(v, "\t", end=" ")
        print()
        current_logs.append(current_log)

    best_sequence = history["sequence_per_play"][
        history["reward_per_play"].index(max(history["reward_per_play"]))]
    best_acc, best_lat = trainer.performance_estimate(best_sequence[0])
    best_reward = trainer.multi_objective_reward(best_acc, best_lat)
    best_result = f"Best sequence: {best_sequence}" \
                  f" \t with accuracy: {round(best_acc, 3)}" \
                  f" \t , latency: {round(best_lat, 3)}" \
                  f" \t and reward: {round(best_reward, 3)}" \
                  f" \t Explored: {len(history['sequence_per_play'])} \t In {int(time.time() - t1)} sec. \n" \
                  f"Total average reward: {np.average(history['reward_per_play'])}"
    print(best_result)

    save_logs(history, current_logs, best_result)
    print("DONE")


def main_naive():
    t1 = time.time()
    cnt_valid = 1
    cnt_skip = 1
    result = False
    history = {
        "accuracy_per_play": [],
        "latency_per_play": [],
        "reward_per_play": [],
        "sequence_per_play": [],
        "sequence_per_episode": [],
        "accuracy_per_episode": [],
        "latency_per_episode": [],
        "reward_per_episode": [],
        "avg_reward_per_episode": [],
        "loss": [],
        "play_counts": [],
        "min_max": [],
        "best_so_far": [],
    }
    current_logs = []
    sequence = []
    reward = 0

    if search_mode == "random":
        watchdog = 0
        while watchdog < naive_timeout:
            watchdog += 1
            sequence = controller.generate_sequence_naive(mode="r") + [list(tokens.keys())[-1]]  # add last layer
            if (sequence in history["sequence_per_play"]) or (not search_space.check_sequence(sequence)):
                cnt_skip += 1
                continue
            architecture = search_space.create_models(samples=[sequence], model_input_shape=model_input_shape)
            if not architecture:
                cnt_skip += 1
                continue

            sequence = sequence[:-1]
            if len(sequence) < max_no_of_layers-1:
                for _ in range((max_no_of_layers-1)-len(sequence)):
                    sequence.append(0)

            accuracy, latency = trainer.performance_estimate(sequence=sequence)
            reward = trainer.multi_objective_reward(accuracy, latency)
            cnt_valid += 1
            history["accuracy_per_play"].append(accuracy)
            history["latency_per_play"].append(latency)
            history["reward_per_play"].append(reward)
            history["sequence_per_play"].append(sequence)
            print(f"Reward: {str(reward)} \t Explored: {cnt_valid} \t Skipped: {cnt_skip} \t Sequence: {sequence}")
            if reward >= naive_threshold:
                result = True
                break

    if search_mode == "bruteforce":
        space = controller.generate_sequence_naive(mode="b")
        for sequence in space:
            sequence = sequence + (list(tokens.keys())[-1],)  # add last layer
            if not search_space.check_sequence(sequence):
                cnt_skip += 1
                continue
            architecture = search_space.create_models(samples=[sequence], model_input_shape=model_input_shape)
            if not architecture:
                cnt_skip += 1
                continue

            accuracy, latency = trainer.performance_estimate(sequence=sequence[:-1])
            reward = trainer.multi_objective_reward(accuracy, latency)
            cnt_valid += 1
            history["accuracy_per_play"].append(accuracy)
            history["latency_per_play"].append(latency)
            history["reward_per_play"].append(reward)
            history["sequence_per_play"].append(sequence)
            print(f"Reward: {str(reward)} \t Explored: {cnt_valid} \t Skipped: {cnt_skip} \t Sequence: {sequence}")
            if reward >= naive_threshold:
                result = True
                break

    if result:
        best_result = f"Best sequence: {sequence} \n" \
                      f"with accuracy: {accuracy} \t latency: {latency} \t and reward: {reward}" \
                      f" \t Explored: {cnt_valid} \t Skipped: {cnt_skip} \t In {int(time.time() - t1)} sec. \n" \
                      f"Total average reward: {np.average(history['reward_per_play'])} " \
                      f"\t Accuracy threshold: {naive_threshold}"
    else:
        best_result = f"No architecture with reward >= {naive_threshold} found. \n" \
                      f"\t Explored: {cnt_valid} \t Skipped: {cnt_skip} \t In {int(time.time() - t1)} sec. \n"

    print(best_result)
    save_logs(history, current_logs, best_result)
    print("DONE")


if __name__ == '__main__':
    print("MODE:", search_mode)
    if search_mode == "rl":
        main_rl()
    else:
        main_naive()
