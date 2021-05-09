# emnas
Embedded Neural Architecture Search

## Checkpoint 1
* Controller is first version with custom loss. 
* Search space generates 382 layer types.

### Notebooks:
#### accuracy_predictor:
fixed_length:
  * search_space: 382 tokens
  * sequence length: 6 with end token
  * model input shape: (5, 383) without end token, 1-hot encoded
  * mse: 0.0204,  mae: 0.0957
  
variable_length: 
  * search_space: 382 tokens
  * sequence length: min: 4, max: 15 with end token
  * model input shape: (15, 383) with end token, 1-hot encoded
  * mse: 0.0188,  mae: 0.0826

#### latency_predictor:
  * search_space: 382 tokens
  * sequence length: 6 with end token
  * model input shape: (5, 383) without end token, 1-hot encoded
  * mse: 38611.683,  mae: 125.655
  * for Sipeed: latency values are measured automatically, using latency_dataset.py
  * for JeVois: latency values are measured manually.

#### reinforcement_learning:
lstm_v1:
* REINFORCE algorithm as custom loss function
* Agent: lstm with fixed input of 5-length sequence
* Input to the lstm: sequence of integers
* output: one discrete probability distribution
* next token is predicted after feeding the lstm with the previous predicted tokens  
* Loss converges, but RL does not increase the reward

lstm_v2:
* input to lstm: 2-D matrix of tokens with their details
* output:
    * v2_1: single probability distribution for the current layer
    * v2_1: one distribution for each layer
    * v2_3: one distribution for each layer parameter

ff_v1:
* REINFORCE algorithm calculates gradient values, applied to the agent as labels.
* Agent: FF with fixed input of 5-length sequence
* Input to the FF: sequence of integers  
* output: one discrete probability distribution
* each episode generates one sequence, each play generates one token

ff_v2:
* Input to the FF: sequence of integers
* output: discrete probability distributions for each layer

ff_v3:
* variable length sequences
* if end token is predicted, length of sequence will be smaller than maximum length
* zero stands for no layer   
* Input to the FF: sequence of integers
* output: discrete probability distributions for each layer
* alpha parameter is small, so agent will not converge. So continuously explores.    
* options to terminate the episode:
  * fixed minimum reward
  * minimum reward set as best average reward so far
* options for initial sequence of the episode:
  * sequence of zeros
  * sequence of random tokens
  * best sequence of previous episode
  * best sequence found so far
* options to define the reward of the next play:
  * reward value of the current play
  * best reward value found so far


09.05.2021
