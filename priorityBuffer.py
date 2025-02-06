import numpy as np
from collections import namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class PriorityBuffer: # specifically adapted to cart pole environment
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha

        # two segment trees involved to find min & sum over ranges
        self.priority_sum = [0 for _ in range(2*self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]
        self.max_priority = 1

        self.data = { # store various arrays for different params
            'state': np.zeros(shape=(capacity, 4), dtype=np.float32),  # CartPole states are 1D vectors of length 4
            'action': np.zeros(shape=capacity, dtype=np.int32),  # Action is an integer
            'reward': np.zeros(shape=capacity, dtype=np.float32),  # Reward is a float
            'next_state': np.zeros(shape=(capacity, 4), dtype=np.float32),  # Next state is also a 1D vector of length 4
            'done': np.zeros(shape=capacity, dtype=bool)  # Boolean indicating whether the episode has ended
        }

        self.next_idx = 0
        self.size = 0 # number of elements within buffer

    def add(self, *args): # add samples
        group = Transition(*args)
        idx = self.next_idx

        self.data['state'][idx] = group.state # store in queues
        self.data['action'][idx] = group.action
        self.data['reward'][idx] = group.reward
        self.data['next_state'][idx] = group.next_state
        self.data['done'][idx] = group.done
        
        self.next_idx = (idx + 1) % self.capacity
        self.size = min(self.capacity, self.size+1)
        priority_alpha = self.max_priority ** self.alpha

        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)
    
    def _set_priority_min(self, idx, priority_alpha):
        idx += self.capacity
        self.priority_min[idx] = priority_alpha

        while idx >= 2:
            idx //= 2 # retrieve parent

            # value of parent is minimum of children
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])
    
    def _set_priority_sum(self, idx, priority):
        idx += self.capacity # begin at leaf
        self.priority_sum[idx] = priority # set leaf priority

        # traverse to root, and update tree
        while idx >= 2:
            idx //= 2

            # make priority sum of children
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2*idx+1]

    def _sum(self): # calaculate sum of sample probabilities - O(1) at root
        return self.priority_sum[1]

    def _min(self):
        return self.priority_min[1]

    # find largest i such that sum of prefix priorities is less than required
    def find_prefix_sum_idx(self, prefix_sum):
        idx = 1
        while idx < self.capacity:
            if self.priority_sum[idx * 2] > prefix_sum: # if sum of left branch higher
                idx = 2 * idx # keep iterating to left branch
            else: # go to right branch & reduce left sum
                prefix_sum -= self.priority_sum[idx*2]
                idx = 2*idx + 1
        return idx - self.capacity

    # sample from priority buffer
    def sample(self, batch_size, beta):
        samples = {
            'weights': np.zeros(shape=batch_size, dtype=np.float32),
            'indexes': np.zeros(shape=batch_size, dtype=np.int32)
        }

        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            samples['indexes'][i] = idx
        
        prob_min = self._min() / self._sum()

        max_weight = (prob_min * self.size) ** (-beta) # beta introduces a bias factor

        for i in range(batch_size):
            idx = samples['indexes'][i]
            prob = self.priority_sum[idx + self.capacity] / self._sum()
            weight = (prob * self.size) ** (-beta)
            samples['weights'][i] = weight / max_weight
        
        for k, v in self.data.items():
            samples[k] = v[samples['indexes']]
        
        return samples

    # update priorities
    def update_priorities(self, indexes, priorities):
        for idx, priority in zip(indexes, priorities):
            self.max_priority = max(self.max_priority, priority)
            priority_alpha = priority ** self.alpha
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    def is_full(self):
        return self.capacity == self.size
    
    def clear(self):
        # Reset the buffer's state
        self.data['transitions'] = [None] * self.capacity  # Reset the transitions
        self.size = 0  # Reset the size
        self.next_idx = 0  # Reset the next index
        self.max_priority = 1  # Reset the max priority

        # Reset priority trees  
        self.priority_sum = [0 for _ in range(2 * self.capacity)]  # Reset sum tree
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]  # Reset min tree