import numpy as np
import tensorflow as tf
import pickle
from sumTree import SumTree

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

MODEL_CACHE = 'Cartpole_DQN_Model.keras'

class Agent:
    def __init__(self, state_size, action_size):
        self.n_actions = action_size
        self.lr = 0.002
        self.gamma = 0.99
        self.exp_prob = 1.0
        self.decay = 0.01
        self.batch_size = 32

        # Replace memory buffer with SumTree for prioritized experience replay
        self.memory_buffer = SumTree(2000)  # Replace with SumTree
        self.max_memory_buffer = 2000
        self.alpha = 0.6
        self.beta = 0.4

        # Sequential model with 2 hidden layers of 24 neurons
        self.model = Sequential([
            Dense(units=32, activation='relu'),
            Dense(units=32, activation='relu'),
            Dense(units=action_size, activation='linear'),
        ])
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr))

    def compute_action(self, current_state):
        current_state = np.array(current_state)
        current_state = np.reshape(current_state, [1, -1])

        if np.random.uniform(0, 1) < self.exp_prob:
            return np.random.choice(range(self.n_actions))
        q_values = self.model.predict(current_state)[0]
        return np.argmax(q_values)

    def update_exp_prob(self):
        self.exp_prob = self.exp_prob * np.exp(-self.decay)
        print(self.exp_prob)

    def store_episode(self, current_state, action, reward, next_state, done):
        priority = 1.0  # Start with maximum priority for new experience
        experience = (current_state, action, reward, next_state, done)
        self.memory_buffer.add(priority, experience)

    def act(self, state):
        """Select action based on epsilon-greedy policy."""
        if np.random.rand() <= self.exp_prob:
            # Exploration: Choose random action
            return np.random.randint(self.n_actions)
        else:
            # Exploitation: Choose action with max Q-value
            state = np.reshape(state, [1, -1])  # Ensure state is in correct shape
            q_values = self.model.predict(state, verbose=0)
            return np.argmax(q_values[0])

    def train(self):
        if self.memory_buffer.size < self.batch_size:
            return

        # Sample batch with priorities
        batch = []
        indices = []
        priorities = []
        total_priority = self.memory_buffer.tree[0]
        
        for _ in range(self.batch_size):
            sample_value = np.random.uniform(0, total_priority)
            index, priority, experience = self._get_sample(sample_value)
            batch.append(experience)
            indices.append(index)
            priorities.append(priority)

        # Compute importance-sampling weights
        sampling_probabilities = np.array(priorities) / total_priority
        weights = (self.memory_buffer.size * sampling_probabilities) ** -self.beta
        weights /= weights.max()

        for i, (current_state, action, reward, next_state, done) in enumerate(batch):
            current_state = np.reshape(current_state, [1, -1])
            next_state = np.reshape(next_state, [1, -1])

            q_current_state = self.model.predict(current_state, verbose=0)
            q_target = reward
            if not done:
                q_target += self.gamma * np.max(self.model.predict(next_state, verbose=0)[0])

            td_error = abs(q_target - q_current_state[0][action])
            self.memory_buffer.update(indices[i], td_error ** self.alpha)

            q_current_state[0][action] = q_target
            self.model.fit(current_state, q_current_state, verbose=0, epochs=1, sample_weight=np.array([weights[i]])) # blocked printout

    def _get_sample(self, value):
        index = 0
        while index < self.memory_buffer.capacity - 1:
            left = 2 * index + 1
            right = left + 1
            if value <= self.memory_buffer.tree[left]:
                index = left
            else:
                value -= self.memory_buffer.tree[left]
                index = right
        data_index = index - self.memory_buffer.capacity + 1
        return index, self.memory_buffer.tree[index], self.memory_buffer.data[data_index]

    def save(self, filename):
        self.model.save(MODEL_CACHE)
        agent_state = {
            'exp_prob': self.exp_prob,
            'memory_buffer': self.memory_buffer,  # Ensure your SumTree class has a proper serialization method
            'decay': self.decay,
            'lr': self.lr,
            'gamma': self.gamma,
            'batch_size': self.batch_size
        }
        with open(f"{filename}_state.pkl", 'wb') as file:
            pickle.dump(agent_state, file)

    @staticmethod
    def load(filename, state_size, action_size):
        model = tf.keras.models.load_model(MODEL_CACHE)
        with open(f"{filename}_state.pkl", 'rb') as file:
            agent_state = pickle.load(file)

        agent = Agent(state_size, action_size)
        agent.model = model
        agent.exp_prob = agent_state['exp_prob']
        agent.memory_buffer = agent_state['memory_buffer']
        agent.decay = agent_state['decay']
        agent.lr = agent_state['lr']
        agent.gamma = agent_state['gamma']
        agent.batch_size = agent_state['batch_size']
        return agent