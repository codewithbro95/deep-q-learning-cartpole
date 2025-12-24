import torch
import torch.optim as optim
import random
import math
import gymnasium as gym
import torch.nn as nn 

from dqn import DQN
from replay_memory import ReplayMemory

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99          # Discount factor (Future rewards are worth slightly less)
EPS_START = 0.9       # Start totally random
EPS_END = 0.05        # End mostly greedy
EPS_DECAY = 1000      # How fast we decay randomness
TAU = 0.005           # Soft update rate for Target Network
LR = 1e-4             # Learning Rate

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 1. Initialize The Brains
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        
        # Sync weights initially
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
        
        self.steps_done = 0

    def select_action(self, state, env):
        """Decide: Random or Brain?"""
        sample = random.random()
        
        # Calculate current threshold
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        
        self.steps_done += 1
        
        if sample > eps_threshold:
            # Exploitation: Ask the Brain
            with torch.no_grad():
                # t.max(1) returns largest column value of each row. 
                # second column on max result is index of where max element was found
                # so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # Exploration: Random Move
            return torch.tensor([[env.action_space.sample()]], dtype=torch.long)

    def optimize_model(self):
            if len(self.memory) < BATCH_SIZE:
                return

            transitions = self.memory.sample(BATCH_SIZE)
            
            # Transpose the batch
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

            # --- THE FIX STARTS HERE ---
            
            # 1. Create a "Mask" of non-final states
            # Returns True if the state is NOT None, False if it is None (Game Over)
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), dtype=torch.bool)
            
            # 2. Create a tensor of ONLY the valid next states
            # We discard the None values here so torch.cat doesn't crash
            non_final_next_states = torch.cat([s for s in batch_next_state if s is not None])

            # --- THE FIX ENDS HERE ---

            state_batch = torch.cat(batch_state)
            action_batch = torch.cat(batch_action)
            reward_batch = torch.cat(batch_reward)
            
            # Calculate Q(s, a)
            q_values = self.policy_net(state_batch).gather(1, action_batch)

            # Calculate V(s_next)
            # Initialize a tensor of Zeros for the next state values
            next_state_values = torch.zeros(BATCH_SIZE)
            
            # Only calculate values for states that actually exist (using the mask)
            # Terminal states stay 0.0, which is correct (no future reward if you are dead)
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

            # Compute the expected Q values
            expected_q_values = reward_batch + (GAMMA * next_state_values)

            # Compute Loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(q_values, expected_q_values.unsqueeze(1))

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.optimizer.step()

            # Soft Update Target Net
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            self.target_net.load_state_dict(target_net_state_dict)

# 1. Init Agent
agent = DQNAgent(state_dim=4, action_dim=2)

# 2. Create some dummy data and fill memory
# We need BATCH_SIZE (128) items to trigger training
dummy_state = torch.tensor([0.0, 0.0, 0.0, 0.0]).unsqueeze(0)

print("Filling memory...")
for _ in range(130):
    agent.memory.push(
        dummy_state, 
        torch.tensor([[0]]), # Action 0
        torch.tensor([1.0]), # Reward
        dummy_state,         # Next State
        False                # Not Done
    )

# 3. Trigger optimization
print("Optimizing...")
try:
    agent.optimize_model()
    print("Optimization step successful! Dimensions match.")
except Exception as e:
    print(f"CRASH: {e}")