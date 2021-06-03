search_space = {
    "mode": "MobileNets",  # normal, MobileNets
    "model_dropout": 0.2,
    "model_loss_function": "categorical_crossentropy",
    "model_optimizer": "Adam",
    "model_lr": 0.001,
    "model_decay": 0.0,
    "model_metrics": ["accuracy"],
}
trainer = {
    "dataset_path_fast": "/home/amirhossein/Codes/Project/Dataset/Dataset_678/dataset_openclose_678_half",
    "dataset_path_full": "/home/amirhossein/Codes/Project/Dataset/Dataset_678/dataset_openclose_678",
    "model_epochs_fast": 8,
    "model_epochs_full": 1,
    "model_validation_split": 0.2,
    "model_batch_size": 10,
    "verbose": 1,
    "multi_obj_weight": 3
}
controller = {
    "max_no_of_layers": 32,
    "agent_lr": 1e-4,
    "min_reward": 0.55,
    "dynamic_min_reward": False,
    "min_plays": 5,
    "max_plays": 20,
    "alpha": 1e-3,  # learning rate in the policy gradient
    "gamma": 0.99,  # decay rate of past observations
    "variance_threshold": 1e-2,
    "valid_actions": True  # True: skips wrong sequences. False: assigns bad reward to wrong sequences
}
predictor = {
    "latency_dataset": "latency_datasets/Dataset_7",
    "search_space_len": 49,
    "no_of_epochs": 400,
    "mode_invalids": "fill",  # fill, ignore
    "mode_predictor": "latency"  # latency, accuracy
}
emnas = {
    "model_output_shape": 2,
    "model_input_shape": (128, 128, 3),
    "search_mode": "rl",  # rl, random, bruteforce
    "naive_threshold": 0.85,
    "naive_timeout": 1e6,
    "no_of_episodes": 1,
    "log_path": "/home/amirhossein/Codes/NAS/emnas/logs"
}
