import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        # Layer 1: Input (State) -> Hidden Layer (128 neurons)
        self.layer1 = nn.Linear(state_dim, 128)
        # Layer 2: Hidden (128) -> Hidden (128)
        self.layer2 = nn.Linear(128, 128)
        # Layer 3: Hidden (128) -> Output (Action Q-Values)
        self.layer3 = nn.Linear(128, action_dim)

    def forward(self, x):
        # Pass through Layer 1 with ReLU activation
        x = F.relu(self.layer1(x))
        # Pass through Layer 2 with ReLU activation
        x = F.relu(self.layer2(x))
        # Pass through Layer 3 (No activation!)
        return self.layer3(x)

# --- Sanity Check ---
# Let's test it immediately. Never assume your dimensions are correct.

# 1. Get dimensions from the environment
state_dim = 4  # [pos, vel, angle, ang_vel]
action_dim = 2 # [left, right]

# 2. Initialize the Brain
net = DQN(state_dim, action_dim)

# 3. Create a dummy state (batch_size=1, features=4)
dummy_state = torch.tensor([0.0, 0.0, 0.0, 0.0]).unsqueeze(0) 

# 4. Get a prediction
q_values = net(dummy_state)

print(f"Input Shape: {dummy_state.shape}")
print(f"Output Shape: {q_values.shape}")
print(f"Raw Q-Values: {q_values}")