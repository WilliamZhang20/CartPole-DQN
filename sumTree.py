import numpy as np

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Sum Tree structure
        self.data = np.zeros(capacity, dtype=object)  # Experience storage
        self.size = 0
        self.write_index = 0  # Tracks where to insert new experiences

    def add(self, priority, data):
        # Calculate the tree index where this new priority should go
        tree_index = self.write_index + self.capacity - 1
        
        # Insert the experience and update the priority in the tree
        self.data[self.write_index] = data  # Overwrite old experience if full
        self.update(tree_index, priority)
        
        # Update the write index in a circular manner
        self.write_index = (self.write_index + 1) % self.capacity
        # Increase the count until full capacity is reached
        if self.size < self.capacity:
            self.size += 1

    def update(self, index, priority):
        # Update the priority at a specific index and propagate the change
        change = priority - self.tree[index]
        self.tree[index] = priority
        self._propagate(index, change)

    def _propagate(self, index, change):
        # Propagate the change up to the root to maintain correct sums
        parent = (index - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
