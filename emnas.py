from controller import Controller
from search_space import SearchSpace
from trainer import Trainer

if __name__ == '__main__':
    search_space = SearchSpace(model_output_shape=2)
    controller = Controller()
    trainer = Trainer()

    tokens = search_space.generate_token()
    samples = controller.generate_sequence(tokens=tokens)
    # manual_samples = [[270, 270, 270, 270, 270, 572]]
    architectures = search_space.create_models(samples=samples, model_input_shape=(128, 128, 3))
    epoch_performance = trainer.train_models(samples=samples, architectures=architectures)
    print("------------------")
    print(epoch_performance)
