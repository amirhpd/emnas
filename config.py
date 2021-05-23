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
    "dataset_path": "/home/amirhossein/Codes/Project/Dataset/Dataset_678/dataset_openclose_678_half",
    "predictor_path": "misc/reinforcement_learning/accuracy_predictor_15.h5",
    "model_validation_split": 0.1,
    "model_batch_size": 10,
    "model_epochs": 1,
    "verbose": 1,
    "hardware": "sipeed",  # sipeed, jevois
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
latency_predictor = {
    "latency": False,  # enable/disable latency
    "latency_dataset": "misc",
    "outlier_limit": {"sipeed": 2000, "jevois": 100},
    "lr": 0.001,
    "train_epochs": 1000,
}
emnas = {
    "model_output_shape": 2,
    "model_input_shape": (128, 128, 3),
    "search_mode": "ff",  # ff, random, bruteforce
    "naive_threshold": 0.65,
    "naive_timeout": 1e6,
    "no_of_episodes": 20,
    "log_path": "/home/amirhossein/Codes/NAS/emnas/logs"
}
