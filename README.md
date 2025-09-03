# DQN：DQN Implementation for CartPole

This is a from-scratch implementation of the Deep Q-Network (DQN) algorithm using PyTorch to solve the CartPole-v1 environment from OpenAI Gym.

## What I Learned
- How to implement the Experience Replay buffer.
- How to build and train a neural network as a Q-function approximator.
- The effect of hyperparameters (like learning rate) on training stability.

## Results
After training for 500 episodes, the agent achieved an average reward of over 195 (out of a maximum of 200) in the CartPole environment.

![Training Progress](/reward_plot.png) # 这里显示你上传的图片

## How to Run the Code
1. Install the required packages: `pip install gym numpy torch`
2. Run the training script: `python dqn.py`

## About Me
I am a graduate student in Computer Science transitioning from a management background. I am passionate about reinforcement learning and explainable AI.
