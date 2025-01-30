import numpy as np
import tensorflow as tf
import gymnasium as gym
import time
from memory import random
from memory import ReplayMemory
import math

import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses TensorFlow INFO and WARNING messages
tf.get_logger().setLevel('ERROR')  # Suppresses logs from the TensorFlow logger

MODEL_CACHE = 'Cartpole_DQN_Model.keras'

class Cartpole_RL_Agent:
    def __init__(self, state_size, action_size):
        self.alpha = 0.001 # learning rate
        self.gamma = 0.999 # discount factor between episodes (from MDPs)
        self.tau = 0.005 # weight of q-network in soft updates (very small for stability)

        # plot array
        self.total_point_history = []

        # Replay memory storage
        self.memory_buffer = ReplayMemory(10000)

        # Q network helps approx a function from state and action to a goodness value
        self.q_network = Sequential([
            Input(shape=state_size),
            Dense(units=64, activation='relu'),
            Dense(units=64, activation='relu'),
            Dense(units=action_size, activation='linear')
        ])

        # Need a target for stable feedback, i.e. loss calculations from a supposed 'actual value'
        self.target_network = Sequential([
            Input(shape=state_size),
            Dense(units=64, activation='relu'),
            Dense(units=64, activation='relu'),
            Dense(units=action_size, activation='linear')
        ])

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha)

    def _compute_loss(self, experiences):
        """
        Calculates Huber Loss
        """
        state, action, next_state, reward, done_val = experiences

        max_qsa = tf.reduce_max(self.target_network(next_state), axis=1)

        y_target = reward + ((1 - done_val)*self.gamma*max_qsa)

        q_values = self.q_network(state)

        q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]), tf.cast(action, tf.int32)], axis=1))
        
        loss_fn = Huber(delta=1.0)  # delta controls the transition point from L2 to L1
        loss = loss_fn(y_target, q_values)
        return loss

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
        if random.random() > epsilon:
            return tf.argmax(q_values, axis=1).numpy()[0]
        else:
            return random.choice([0, 1])

    def _sample_memory(self, batch_size):
        transitions = self.memory_buffer.sample(batch_size)
        # tensorflow operations will work on gpu rather than numpy CPU-based
        states = tf.stack([tf.convert_to_tensor(e.state, dtype=tf.float32) for e in transitions if e is not None], axis=0)
        actions = tf.stack([tf.convert_to_tensor(e.action, dtype=tf.float32) for e in transitions if e is not None], axis=0)
        rewards = tf.stack([tf.convert_to_tensor(e.reward, dtype=tf.float32) for e in transitions if e is not None], axis=0)
        next_states = tf.stack([tf.convert_to_tensor(e.next_state, dtype=tf.float32) for e in transitions if e is not None], axis=0)
        done_vals = tf.stack([tf.convert_to_tensor(e.done, dtype=tf.float32) for e in transitions if e is not None], axis=0)
        return (states, actions, next_states, rewards, done_vals)

    def train_episodes(self, num_episodes):
        start = time.time()
        self.total_point_history = []
        env = gym.make('CartPole-v1')

        episode_rec = 10
        max_iters = 1000
        batch_size = 64
        e_decay = 0.98
        e_min = 0.001 # end with a nearly zero chance of exploration
        epsilon = 1.0

        # update every few episodes to minimize computations
        update_interval = 4

        self.memory_buffer.clear()

        for i in range(num_episodes):
            curr_state, _ = env.reset()
            total_points = 0
            for t in range(max_iters):
                # transform into q_network input
                state_qn = np.expand_dims(curr_state, axis=0)
                q_values = self.q_network.predict(state_qn, verbose=0)

                # get action by random choice between explore vs exploit
                action = self.get_action(q_values, epsilon)

                # take next action - retrieve environ reward
                next_state, reward, done, _ , _ = env.step(action)

                # record in memory
                self.memory_buffer.push(curr_state, action, next_state, reward, done)

                do_update= ( (t + 1) % update_interval == 0) and ( self.memory_buffer.__len__() > batch_size )

                if do_update:
                    # compute soft update + update the agent... 
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
        
        self.plot_history() # show a plot after training!

    def plot_history(self):
        # from the element `total point history`
        plt.plot(range(len(self.total_point_history)), self.total_point_history)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Learning Progress Over Time')
        plt.savefig('learning_progress.png')  # Save the plot as an image file
        plt.show()  # Display the plot

    def load_model(self):
        self.q_network = tf.keras.models.load_model(MODEL_CACHE)
        self.target_network = self.q_network