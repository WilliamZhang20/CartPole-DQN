# Cart Pole DQN

In classic Q-learning, a Q-table was used to store the expected utility values for each state-action pair.
But for more complex examples such as a [cart pole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) control environment, it is more optimal to use a neural network to map states, and actions to a reward. 

This is called a **Deep Q Network (DQN)**, as described in [this](https://arxiv.org/abs/1312.5602) very famous paper setting out the concept.

In this repository, I have trained a DQN to control a cart pole and keep the pole vertical. This was done in the Gymnasium (formerly OpenAI gym) simulation environment, and the agent's training was written in Python using Tensorflow. 

## Overview

The file `agent.py` contains methods to train and execute actions for the cart pole controller agent, and the file `train_cartpole.py` runs training cycles and captures a video of the trained agent in action! The folder `cartpole-video` contains a (too) short video of the agent balancing the pole. 

Finally, file `requirements.txt` has a list of libraries needed not only for training, but also for capturing videos.

Effective training can take a long time, up to an hour. During that process, one can observe that the agent is progressively able to balance the pole for longer periods of time, although occasionally, it makes a mistake and the game terminates early.

The result is that the total performance of the agent will increase gradually, but when it reaches the highest set point, its performance will oscillate, as shown in the image `rewards_result.png`. 

The video in the folder `cartpole-video` is of an agent playing a single game of the cartpole, until it breaks the environment's requirements (see first sentence of "How it works"). 

So far, the maximum time it can do this is 7 seconds, and the learning progression is still unstable. 

Demonstration (A 5-second long GIF):

![eval-episode-0](https://github.com/user-attachments/assets/d1386d34-112d-468e-bf0e-d99466100c0e)


## How it works

The overall goal of the training process is to make sure that it holds the pole vertical for as long as possible. If the pole's angle exceeds 12 degrees, or the cart veers out of the 'video frame', then the game is over. 

In the process, I set a maximum duration, which the DQN attempts to reach, and each time it does not, a reward of -1 is assigned. At each training round, it processes a set number of episodes (currently hardcoded to 300), and plays the game. At each training step, it store the rewards in a memory buffer, which uses Prioritized Experience Replay (PER) to sample later on. The PER is implemented using a Sum Tree data structure, in which the most valuable experiences are collected.

That deep neural network model is used at each step of the game to determine moves, which can either be pushing the cart left or right. It contains two fully connected hiddden layers, and outputs the Q-values for each possible action in the current state.

Additionally, after training, the agent class's data members, including its neural network model, are stored in a serialized `.pkl` file, which can then be pulled to make videos without having to redo the training. 

## Credits

The original code is from [this](https://towardsdatascience.com/deep-q-networks-theory-and-implementation-37543f60dd67) Medium blog, to which I added adjustments to the new version of Gymnasium, and other improvements, mentioned above.
