search_space = {
    "model_dropout": 0.2,
    "model_loss_function": "categorical_crossentropy",
    "model_optimizer": "Adam",
    "model_lr": 0.001,
    "model_decay": 0.0,
    "model_metrics": ["accuracy"],
}
trainer = {
    "dataset_path": "/home/amirhossein/Codes/Project/Dataset/Dataset_678/dataset_openclose_678_half",
    "model_validation_split": 0.1,
    "model_batch_size": 10,
    "model_epochs": 3,
    "verbose": 0,
    "hardware": "sipeed",  # sipeed, jevois
}
controller = {
    "no_of_samples_per_epoch": 10,
    "no_of_layers": 12,
    "rnn_dim": 100,
    "rnn_lr": 0.01,
    "rnn_decay": 0.1,
    "rnn_no_of_epochs": 200,
    "rnn_loss_alpha": 0.9,
    "rl_baseline": 0.85,
    "latency_coefficient": 0.5,
    "verbose": 0,
}
latency_predictor = {
    "latency": False,  # enable/disable latency
    "latency_dataset": "misc",
    "outlier_limit": {"sipeed": 2000, "jevois": 100},
    "lr": 0.001,
    "train_epochs": 1000,
}
emnas = {
    "no_of_nas_epochs": 10,
    "model_output_shape": 2,
    "model_input_shape": (128, 128, 3),
    "search_mode": "rnn",  # rnn, random, bruteforce
    "naive_threshold": 0.8,
    "naive_timeout": 1e6,
}
