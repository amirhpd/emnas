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
  
4. After these steps are done, the main search procedure can be executed several times by running the executable script [emnas\.py](emnas.py). Several hyper-parameters select and adjust the behaviors of the search system. All adjustable hyper-parameters are saved in a single file called [config\.py](config.py). 

Current repository contains a search space that is inspired from [MobileNets](https://arxiv.org/abs/1704.04861)
and contains predictor model files that are already built and trained on [Dataset_7](Dataset_7/table.csv).

### UML Diagram
![alt text](https://github.com/amirhpd/emnas/media/overall_uml.png "Overall UML")

