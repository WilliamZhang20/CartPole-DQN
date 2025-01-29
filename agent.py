import numpy as np
import tensorflow as tf
import gymnasium as gym
import time
import memory
import math

import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam

MODEL_CACHE = 'Cartpole_DQN_Model.keras'

class Cartpole_RL_Agent:
    def __init__(self, state_size, action_size):
        self.alpha = 0.001 # learning rate
        self.gamma = 0.999 # discount factor between episodes (from MDPs)
        self.tau = 0.005 # weight of q-network in soft updates (very small for stability)

        # plot array
        self.total_point_history = []
        
        # Replay memory storage
        self.memory_buffer = memory.ReplayMemory(10000)

        # Q network helps approx a function from state and action to a goodness value
        self.q_network = Sequential([
            Input(shape=state_size),
            Dense(units=64, activation='relu'),
            Dense(units=64, actiavtion='relu'),
            Dense(units=action_size, activations='linear')
        ])
        
        # Need a target for stable feedback, i.e. loss calculations from a supposed 'actual value'
        self.target_network = Sequential([
            Input(shape=state_size),
            Dense(units=64, activation='relu'),
            Dense(units=64, actiavtion='relu'),
            Dense(units=action_size, activations='linear')
        ])

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha)
    
    def _compute_loss(self, experiences):
        """
        Calculates MSE
        """
        state, action, reward, next_state, done_val = experiences

        max_qsa = tf.reduce_max(self.target_net(next_state), axis=1)

        y_target = reward + ((1 - done_val)*self.gamma*max_qsa)
        
        q_values = self.q_network(state)

        q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]), tf.cast(action, tf.int32)], axis=1))
        loss = MSE(y_target, q_values)
        return loss
    
    @tf.function # turn on computation graph construction!
    def _agent_learn(self, experiences):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(experiences)
        
        gradients = tape.gradient(loss, self.q_network.trainable_variables)

        # update weights in Adam
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

        self._update_target_network()

    def _update_target_network(self):
        # Use soft updates to updates over the parameters of the target network
        # Formula w_target θ_newt =  τ * θ_q_net + (1 - τ) * θ_old

        for target_weights, q_net_weights in zip(
            self.target_network.weights, self.q_network.weights
        ):
            target_weights.assign(self.tau * q_net_weights + (1.0 - self.tau) * target_weights)

    def get_action(self, q_values, epsilon=0.0):
        # use ε-greedy policy
        if memory.random.random() > epsilon:
            return np.argmax(q_values.numpy()[0])
        else:
            return memory.random.choice(np.arange(4))

    def _sample_memory(self, batch_size):
        transitions = self.memory_buffer.sample(batch_size)
        states = tf.convert_to_tensor(
            np.array([e.state for e in transitions if e is not None]), dtype=tf.float32
        )
        actions = tf.convert_to_tensor(
            np.array([e.action for e in transitions if e is not None]), dtype=tf.float32
        )
        rewards = tf.convert_to_tensor(
            np.array([e.reward for e in transitions if e is not None]), dtype=tf.float32
        )
        next_states = tf.convert_to_tensor(
            np.array([e.next_state for e in transitions if e is not None]), dtype=tf.float32
        )
        done_vals = tf.convert_to_tensor(
            np.array([e.done for e in transitions if e is not None]), dtype=tf.float32
        )
        return (states, actions, next_states, rewards, done_vals)

    def train_episodes(self, num_episodes):
        start = time.time()
        self.total_point_history = []
        env = gym.make('CartPole-v1')

        episode_rec = 100
        max_iters = 1000
        batch_size = 64
        e_decay = 0.99
        e_min = 0.01
        epsilon = 1.0

        self.memory_buffer.clear()

        for i in range(num_episodes):
            curr_state, _ = env.reset()
            total_points = 0
            for t in range(max_iters):
                # transform into q_network input
                state_qn = np.expand_dims(curr_state, axis=0)
                q_values = self.q_network(state_qn)

                # get action by random choice between explore vs exploit
                action = self.get_action(q_values, epsilon)

                # take next action - retrieve environ reward
                next_state, reward, done, _ , _ = env.step(action)

                # record in memory
                self.memory_buffer.append(memory.Transition(curr_state, action, next_state, reward, done))

                # complete soft update every episode...
                experiences = self._sample_memory(batch_size)

                self._agent_learn(experiences)

                curr_state = next_state.copy() # copy over numpy array
                total_points += reward    

                if done:
                    break
            
            self.total_point_history.append(total_points)
            epsilon = max(e_min, e_decay * epsilon)
            avg_latest_points=np.mean(self.total_point_history[-episode_rec:])
            
            if (i+1) % episode_rec == 0:
                print(f"\rEpisode {i+1} | Total point average of the last {episode_rec} episodes: {avg_latest_points:.2f}")

        time_taken = time.time() - start
        print(f"\nTotal Runtime: {time_taken:.2f} s ({(time_taken/60):.2f} min)")
        self.q_network.save(MODEL_CACHE)

    def plot_history(self):
        # from the element `total point history`
        plt.plot(range(len(self.total_point_history)), self.total_point_history)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Learning Progress Over Time')
        plt.savefig('learning_progress.png')  # Save the plot as an image file
        plt.show()  # Display the plot

    @staticmethod
    def load_model(self):
        self.q_network = tf.keras.models.load_model(MODEL_CACHE)
        self.target_network = self.q_network