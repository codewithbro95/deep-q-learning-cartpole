import random
from collections import deque

class ReplayMemory:
    def __init__(self, capacity):
        # A queue that automatically pops old items when full
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Save a transition"""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Pick a random batch of transitions"""
        # random.sample is fast and does exactly what we need
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- Sanity Check ---
# 1. Create memory with capacity for 5 items
mem = ReplayMemory(capacity=5)

# 2. Push 10 items (Note: the first 5 should be auto-deleted!)
for i in range(10):
    mem.push(state=i, action=i, reward=i, next_state=i+1, done=False)

# 3. Check length (Should be 5, not 10)
print(f"Memory Length: {len(mem)}")

# 4. Sample a batch of 3
batch = mem.sample(3)
print(f"Random Batch Sample: {batch}")