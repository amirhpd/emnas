search_space = {
    "model_dropout": 0.2,
    "model_loss_function": "categorical_crossentropy",
    "model_optimizer": "Adam",
    "model_lr": 0.001,
    "model_decay": 0.0,
    "model_metrics": ["accuracy"],
}
trainer = {
    "dataset_path": "/home/amirhossein/Codes/Project/Dataset/Dataset_678/dataset_openclose_678",
    "model_validation_split": 0.1,
    "model_batch_size": 10,
    "model_epochs": 1,
    "verbose": 0,
}
controller = {
    "no_of_samples_per_epoch": 5,
    "no_of_layers": 6,
    "rnn_dim": 200,
    "rnn_lr": 0.01,
    "rnn_decay": 0.1,
    "rnn_no_of_epochs": 200,
    "rnn_loss_alpha": 0.9,
    "rl_baseline": 0.85,
    "verbose": 0,
}
emnas = {
    "no_of_nas_epochs": 20,
    "model_output_shape": 2,
    "model_input_shape": (128, 128, 3),
}