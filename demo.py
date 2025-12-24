import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import pygame  # We need this to capture your key presses
import numpy as np

# --- 1. Define the Architecture ---
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
# render_mode="human" creates the window we need
env = gym.make("CartPole-v1", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Load the brain
policy_net = DQN(state_dim, action_dim)
try:
    policy_net.load_state_dict(torch.load("cartpole_best.pth", weights_only=True))
    print("Loaded 'cartpole_best.pth' successfully.")
except FileNotFoundError:
    print("Error: 'cartpole_best.pth' not found. Make sure you trained and saved it!")
    exit()

policy_net.eval()

print("\n" + "="*40)
print("     INTERACTIVE DEMO MODE")
print("="*40)
print(" The AI is trying to balance.")
print(" YOU are the chaos.")
print(" -> Press LEFT ARROW to shove the pole left.")
print(" -> Press RIGHT ARROW to shove the pole right.")
print(" -> Close the window to exit.")
print("="*40 + "\n")

# --- 3. The Interactive Loop ---
while True:  # Infinite loop for continuous play
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    
    # We loop until the pole falls (terminated)
    while True:
        # --- A. HANDLE USER INTERACTION (The Shove) ---
        # We check PyGame events to see if you pressed a key
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()
            
            if event.type == pygame.KEYDOWN:
                # Get the current internal state of the physics engine
                # State = [cart_pos, cart_vel, pole_angle, pole_vel]
                current_sim_state = list(env.unwrapped.state)
                
                # Apply a "Kick" to the pole velocity (index 3)
                if event.key == pygame.K_LEFT:
                    print(">>> KICK LEFT! <<<")
                    current_sim_state[3] -= 0.3  # Adjust this number to kick harder/softer
                    current_sim_state[2] -= 0.1  # Slight angle tilt to make it visual immediately
                    
                elif event.key == pygame.K_RIGHT:
                    print(">>> KICK RIGHT! <<<")
                    current_sim_state[3] += 0.3
                    current_sim_state[2] += 0.1
                
                # Force the physics engine to accept our hacked state
                env.unwrapped.state = np.array(current_sim_state)

        # --- B. AI MAKES DECISION ---
        # Note: We re-read the state from env because we might have just hacked it above
        # We need to manually convert the numpy state to tensor for the AI
        current_observation = torch.tensor(env.unwrapped.state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            # AI decides based on the (potentially kicked) state
            action = policy_net(current_observation).max(1)[1].view(1, 1)

        # --- C. STEP THE PHYSICS ---
        observation, reward, terminated, truncated, info = env.step(action.item())
        
        # Check if the pole fell
        if terminated:
            print(">> It fell! Resetting...")
            break # Break inner loop to reset env
        
        # If it hits 500 steps (truncated), we just reset to keep it going
        if truncated:
            print(">> Max steps reached (Win). Resetting...")
            break
            
        # Update state for next loop iteration
        state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)