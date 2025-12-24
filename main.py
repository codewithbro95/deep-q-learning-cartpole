import matplotlib.pyplot as plt
import torch
from dqn_agent import DQNAgent
import gymnasium as gym

from live_visualizer import LiveVisualizer

# 1. Setup the Environment
env = gym.make("CartPole-v1") # No render_mode="human" yet to speed up training

# 2. Init the Agent
# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

agent = DQNAgent(n_observations, n_actions)

# --- 1. INITIALIZE VISUALIZER ---
viz = LiveVisualizer(agent.policy_net)

# 3. Tracking metrics
episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration (Steps)')
    plt.plot(episode_durations)
    plt.pause(0.001)  # pause a bit so that plots are updated

# --- THE LOOP ---
print("Starting Training...")
num_episodes = 600 
best_score = 0 # For tracking the best score so we know when to save the model

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device='cpu').unsqueeze(0)
    
    for t in range(1000): # Run for max 1000 steps (or until it drops)
        # 1. Select Action
        action = agent.select_action(state, env)
        
        # 2. Step the Env
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device='cpu')
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device='cpu').unsqueeze(0)

        # 3. Store in Memory
        # Note: We store "done" as an integer (0 or 1) because the ReplayMemory expects it
        agent.memory.push(state, action, reward, next_state, done)

        # 4. Move to the next state
        state = next_state

        # 5. Optimize (Learn!)
        agent.optimize_model()

        # If the pole fell (terminated) or time ran out (truncated)
        if done:
            episode_durations.append(t + 1)
            current_score = t + 1
            
            # --- NEW CODE: SAVE BEST MODEL ---
            # We define a "high score" variable outside the loop initially (e.g., best_score = 0)
            if current_score > best_score:
                best_score = current_score
                torch.save(agent.policy_net.state_dict(), "cartpole_best.pth")
                print(f"New High Score: {best_score} (Saved model)")
            # ---------------------------------
            # --- 2. UPDATE VISUALIZER (Once per episode) ---
            viz.update()
            break

    # Print progress every 20 episodes
    if i_episode % 20 == 0:
        print(f"Episode {i_episode} | Score: {episode_durations[-1]} | Epsilon: {agent.steps_done}")

print("Complete")
plot_durations(show_result=True)
plt.show()