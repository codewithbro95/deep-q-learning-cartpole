import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# --- 1. Define the Architecture (Must match exactly!) ---
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# --- 2. Setup ---
env = gym.make("CartPole-v1", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Load the brain
policy_net = DQN(state_dim, action_dim)
# Load the weights we saved earlier
policy_net.load_state_dict(torch.load("cartpole_best.pth", weights_only=True))
policy_net.eval() # Set to evaluation mode (turns off dropout/batchnorm if we had them)

# --- 3. The Game Loop ---
for episode in range(5): # Let's watch 5 games
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    total_reward = 0
    
    print(f"Starting Episode {episode+1}...")
    
    while True:
        # No random actions! We ask the network directly.
        with torch.no_grad():
            action = policy_net(state).max(1)[1].view(1, 1)

        # Take the action
        observation, reward, terminated, truncated, info = env.step(action.item())
        total_reward += reward

        # Render is automatic because we set render_mode="human"
        
        # Prepare next state
        if terminated or truncated:
            print(f"Episode Finished. Score: {total_reward}")
            break
            
        state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        
        # Optional: Slow it down slightly so human eyes can track it
        # time.sleep(0.01) 

env.close()