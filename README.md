# Deep Q-Learning (DQN) for CartPole-v1

This repository contains a robust implementation of a **Deep Q-Network (DQN)** agent capable of solving the classic *CartPole-v1* control problem.

Built from scratch using **PyTorch** and **Gymnasium**, this project demonstrates key Reinforcement Learning concepts including Experience Replay, Target Networks, and Epsilon-Greedy exploration. It also features a custom interactive demo mode that allows to stress-test the trained agent by applying external forces.

## Overview

**The Goal:** Balance a pole on a cart by moving the cart left or right.
**The Challenge:** The environment provides no labeled data. The agent must learn solely through trial and error, associating actions with delayed rewards.

### Key Features

* **Deep Q-Network (DQN):** Replaces the traditional Q-Table with a Neural Network to handle continuous state spaces.
* **Experience Replay:** Uses a circular buffer to store and sample past transitions, breaking temporal correlations and stabilizing training.
* **Target Network:** Implements a secondary "frozen" network to calculate stable Q-value targets, preventing oscillation.
* **Robust Checkpointing:** Automatically saves the *best* performing model (not just the last one) to avoid catastrophic forgetting.
* **Interactive Demo:** A `pygame`-based inference script that allows humans to "kick" the cart to test the agent's recovery reflexes.

## 🛠️ Installation

1. **Clone the repository:**

```bash
git clone https://github.com/codewithbro95/deep-q-learning-cartpole.git
cd deep-q-learning-cartpole

```

1. **Install dependencies:**
It is recommended to use a virtual environment.

```bash
pip install gymnasium[classic_control] torch matplotlib pygame numpy

```

## 📂 File Structure

| File | Description |
| --- | --- |
| `main.py` | The main entry point for **training**. Contains the training loop and performance plotting. |
| `dqn_agent.py` | Contains the `DQNAgent` class, `ReplayMemory`, and the `DQN` neural network architecture. |
| `demo.py` | The **inference** script. Loads the trained model and runs the interactive simulation. |
| `cartpole_best.pth` | The saved weights of the best-performing model (generated after training). |

## 🚀 Usage

### 1. Training the Agent

To train the model from scratch, run:

```bash
python main.py

```

* **What happens:** The agent will play 600 episodes.
* **Visuals:** A live plot will appear showing the duration (score) of each episode.
* **Output:** The script will save the best model weights to `cartpole_best.pth` whenever a new high score is reached.

### 2. Running the Interactive Demo

Once you have a trained model (or if you are using the provided weights), run:

```bash
python demo.py

```

![Alt text](artifacts/demo_ss.png?raw=true "demo_ss")

Controls:

* **LEFT ARROW:** Apply a sudden force (kick) to the left.
* **RIGHT ARROW:** Apply a sudden force (kick) to the right.

* **Observation:** Watch how the agent frantically adjusts the cart position to recover the pole's balance after being shoved.

## Architecture

### The Neural Network ("The Brain")

We use a simple Multi-Layer Perceptron (MLP) to approximate the Q-Value function:

* **Input Layer:** 4 Neurons (Cart Position, Cart Velocity, Pole Angle, Pole Velocity)
* **Hidden Layer 1:** 128 Neurons (ReLU activation)
* **Hidden Layer 2:** 128 Neurons (ReLU activation)
* **Output Layer:** 2 Neurons (Action Left, Action Right) - *Linear activation (Raw Q-Values)*

### Hyperparameters

* `BATCH_SIZE`: 128
* `GAMMA` (Discount Factor): 0.99
* `EPSILON` (Exploration): Decays from 0.9 to 0.05
* `LEARNING_RATE`: 1e-4 (AdamW Optimizer)

## Learning

![Alt text](artifacts/learnin_rate.png?raw=true "demo_ss")

## 🐛 Troubleshooting

* **"Crash: NoneType Error"**: This usually happens if the training loop tries to process the "Next State" after the game has ended. The code handles this by masking out terminal states in the `optimize_model` function.
* **Model Performance Degrades**: If the model performs well at episode 300 but fails at episode 600, this is "Catastrophic Forgetting." Ensure you are using `cartpole_best.pth` (the checkpointed model) and not the final state of the network.

*Project created for educational purposes to understand Deep Reinforcement Learning engineering practices under the hood.*
