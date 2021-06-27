## Embedded Neural Architecture Search (EMNAS)

This is the code repository for master's thesis: <br />
**Intelligent Neural Network Design for Robotic Embedded Vision**

EMNAS automatically designs CNN architectures for a specific hardware platform and a specific task. <br />
For the current implementation, the hardware platform is [Sipeed Maix Bit](https://www.seeedstudio.com/Sipeed-MAix-BiT-for-RISC-V-AI-IoT-p-2872.html) smart camera,
and the task is robotic grasp verification.

### Initialization
Following steps are needed for the first initialization:
1. Define the parameters of the search space: <br />
  The ranges that are required to establish the search space should be saved in a JSON file [search_space_mn\.json](search_space_mn.json).
  
2. Create the dataset of accuracy and latency values: <br />
  Two executable scripts are responsible for this reason. [latency_dataset\.py](latency_dataset.py) generates the CNN models and measures latency values. Then [accuracy_dataset\.py](accuracy_dataset.py) reads models and measures the accuracy values.
  
3. Build and train the predictors: <br />
  The executable script  [predictor\.py](predictor.py) is responsible for reading the dataset, building the regression models, training, and saving them on the disk.
  
4. After these steps are done, the main search procedure can be executed several times by running the executable script [emnas\.py](emnas.py). Several hyper-parameters select and adjust the behaviors of the search system. All adjustable hyper-parameters are saved in a single file: [config\.py](config.py). 

Current repository contains a search space that is inspired from [MobileNets](https://arxiv.org/abs/1704.04861)
and contains predictor model files that are already built and trained on [Dataset_7](https://github.com/amirhpd/emnas/blob/master/latency_datasets/Dataset_7/table.csv).

### UML Diagram
![alt text](https://github.com/amirhpd/emnas/blob/master/media/overall_uml.png "Overall UML")

### Modules
* [Search space](search_space_mn.py) class is responsible for generating the token dictionary.
* The Search strategy functionalities are implemented in the [Controller](controller.py) class. EMNAS follows the RL approach for search strategy.
* [Trainer](trainer.py) returns the measured or estimated reward of an architecture.
* In order to measure the latency of a CNN, it should be executed on the hardware. Class [CameraDrive](camera_drive.py) establishes the connection between the host computer that runs  EMNAS and the hardware.
* [latency_dataset](latency_dataset.py) generates a dataset of random CNN models and measures their latency.
* [accuracy_dataset](accuracy_dataset.py) reads the dataset, trains the CNN models and measures their accuracy.
* Class [Predictor](predictor.py) together with an executable script read the dataset and trains the NN models.
* The main NAS loop will be executed by running the [emnas](emnas.py) script. EMNAS can perform 3 modes of search: RL-based NAS, random search, and brute-force search.

### Configurations
All adjustable parameters are collected in a single file: [config\.py](config.py).
Hyper-parameters are categorized into 5 groups according to the script that calls them:
#### Search space
* *mode*: Selects between two possibilities of network topology: normal, MobileNets
* *model_dropout*: Dropout value for models with normal topology
* *model_loss_function, model_optimizer, model_lr, model_decay, model_metrics*: Compile parameters of the sequence as a part of converting the sequence to a Keras object.
#### Trainer
* *dataset_path_fast*: Path to the classification dataset, when training mode is set to fast.
* *dataset_path_full*: Path to the classification dataset, when training mode is set to full.
* *model_epochs_fast*: Number of epochs to train a sequence, fast mode.
* *model_epochs_full*: Number of epochs to train a sequence, full mode.
* *model_validation_split, model_batch_size, verbose*: Parameters for training and validating sequences.
* *objective_type*: Selects between two available multi-objective formulas. 
* *multi_obj_parameter*: If *objective_type = type_1*: ratio of accuracy to latency. If *objective_type = type_2*: latency threshold in milliseconds.
#### Controller
* *max_no_of_layers*: Defines the maximum possible number of layers in the sequence. It defines the input shape of the NN agent. It also affects the size of search space. 
* *agent_lr*: Learning rate of the NN agent. Larger values of this parameter lead to early convergence of the agent. 
* *min_reward, dynamic_min_reward*: They define the criteria to episode termination. Episodes are terminated if the reward value of last play drops blow the *min_reward*. <br />
  *dynamic_min_reward = False*: *min_reward* is fixed to the value that is defined here. <br /> 
  *dynamic_min_reward = True*: *min_reward* is set as the best reward average. 
* *min_plays, max_plays*: Other criteria to define the termination of episodes. The episode cannot be terminated until minimum number of plays are done. If no factor terminates the episode, it terminates after maximum number of plays are done.
* *alpha*: Learning rate of the policy gradient. It is a part of REINFORCE formulation.
* *gamma*: Decay rate of past observations. It is a part of REINFORCE formulation.
* *variance_threshold*: A part of episode termination criteria. If reward values of plays in episode stay similar to each other, the episode is terminated. Degrees of similarity is defined by *variance_threshold*. Larger numbers of this parameter allows more variation in rewards before episode termination. 
* *valid_actions*: Defines the strategy to deal with invalid sequences. <br />
  *True*: skips invalid sequences. <br />
  *False*: assigns bad reward to invalid sequences.
#### Predictor
* *prediction_dataset*: Path to the dataset that is created for training predictors.
* *search_space_len*: Length of the search space with which the dataset is generated. It is equal to the last token of the sequences in the dataset.
* *no_of_epochs*: For training the latency and accuracy predictors.
* *mode_invalids*: Defines how to deal with the items that belong to too-large sequences. <br /> 
  *fill*: assigns a high latency and low accuracy value to too-large sequences. <br />
  *ignore*: removes too-large sequences from the dataset.
* *mode_predictor*: Selects to train latency predictor or accuracy predictor.
#### emnas
* *model_output_shape*: Defines the number of outputs for the CNN architectures that are generated by EMNAS. It also defines the output activation function. For *model\_output\_shape = 1*, activation function is *Sigmoid*. For values larger than 1, activation function is *Softmax*.
* *model_input_shape*: Defines the input image shape for the CNN architectures that are generated by EMNAS.
* *search_mode*: Selects between the search modes: reinforcement-learning, random, brute-force.
* *naive_threshold*: In the case of random and brute-force modes, the search procedure is terminated when a sequence with reward grater than *naive_threshold* is found.
* *naive_timeout*: In the case of random and brute-force modes, if *naive_threshold* is not satisfied after a number of iterations defined by *naive_timeout*, the search is terminated with failure. 
* *no_of_episodes*: Defines the number of NAS episodes in reinforcement-learning mode.
* *log_path*: Path to save logs on disk.









