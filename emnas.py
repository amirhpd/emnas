from controller import Controller
from search_space import SearchSpace
from trainer import Trainer
import time

if __name__ == '__main__':
    no_of_nas_epochs = 4
    t1 = time.time()
    search_space = SearchSpace(model_output_shape=2)
    tokens = search_space.generate_token()
    controller = Controller(tokens=tokens)
    trainer = Trainer()

    bests = []
    for nas_epoch in range(no_of_nas_epochs):
        print(f"NAS epoch {nas_epoch+1}/{no_of_nas_epochs} ----------")
        samples = controller.generate_sequence_random_predict()
        architectures = search_space.create_models(samples=samples, model_input_shape=(128, 128, 3))
        print("Candidates:")
        print(samples)
        print("Training architectures ----------")
        epoch_performance = trainer.train_models(samples=samples, architectures=architectures)
        best_of_epoch = [j for j in epoch_performance if j[1] == max([i[1] for i in epoch_performance])][0]
        bests.append(best_of_epoch)
        print("Best architecture:")
        print(best_of_epoch)
        print("---------------------------------------------")
        print("---------------------------------------------")

    best = [j for j in bests if j[1] == max([i[1] for i in bests])][0]
    print("Best architecture:")
    print(best)
    best_translated = search_space.translate_sequence(best[0])
    print(best_translated)
    print("DONE", time.time() - t1)


