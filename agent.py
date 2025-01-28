import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam

MODEL_CACHE = 'Cartpole_DQN_Model.keras'

class Agent:
    def __init__(self, state_size, action_size):
        self.alpha = 0.001 # learning rate
        self.memory_size = 10000 # replay memory
        self.gamma = 0.999 # discount factor between episodes (from MDPs)
        self.tau = 0.005 # weight of q-network in soft updates (very small for stability)

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
    def agent_learn(self, experiences):
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

    def train_episodes(self, num_episodes):