from controller import Controller
from search_space import SearchSpace
from trainer import Trainer
import time

if __name__ == '__main__':
    mode = "r"
    threshold = 0.8
    timeout = 1e6

    t1 = time.time()
    search_space = SearchSpace(model_output_shape=2)
    tokens = search_space.generate_token()
    controller = Controller(tokens=tokens)
    trainer = Trainer()
    result = None
    cnt_valid = 1
    cnt_skip = 1

    if mode == "b":
        space = controller.generate_sequence_naive(mode=mode)  # brute-force
        for sequence in space:
            if not controller.check_sequence(sequence):
                cnt_skip += 1
                continue
            cnt_valid += 1
            architecture = search_space.create_model(sequence=sequence, model_input_shape=(128, 128, 3))
            print("Candidate:", sequence)
            epoch_performance = trainer.train_models(samples=[sequence], architectures=[architecture])
            print("Accuracy:", epoch_performance[0][1])
            if epoch_performance[0][1] >= threshold:
                result = epoch_performance
                break

    if mode == "r":
        watchdog = 0
        history = []
        while watchdog < timeout:
            watchdog += 1
            sequence = controller.generate_sequence_naive(mode=mode)  # random
            if (sequence in history) or (not controller.check_sequence(sequence)):
                cnt_skip += 1
                continue
            cnt_valid += 1
            history.append(sequence)
            architecture = search_space.create_model(sequence=sequence, model_input_shape=(128, 128, 3))
            print("Candidate:", sequence)
            epoch_performance = trainer.train_models(samples=[sequence], architectures=[architecture])
            print("Accuracy:", epoch_performance[0][1])
            if epoch_performance[0][1] >= threshold:
                result = epoch_performance
                break

    t = time.time() - t1
    if result:
        print("Found architecture:")
        print(search_space.translate_sequence(result[0][0]))
        print(f"With accuracy: {result[0][1]} after checking {cnt_valid} sequences and skipping {cnt_skip} sequences.")
        print(f"DONE (t:{t})")
    else:
        print(f"No architecture with accuracy >= {threshold} found.")
        print(f"DONE (t:{t})")





