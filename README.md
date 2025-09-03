[DQN.py](https://github.com/user-attachments/files/22122118/DQN.py)# DQN：DQN Implementation for CartPole

This is a from-scratch implementation of the Deep Q-Network (DQN) algorithm using PyTorch to solve the CartPole-v1 environment from OpenAI Gym.

## What I Learned
- How to implement the Experience Replay buffer.
- How to build and train a neural network as a Q-function approximator.
- The effect of hyperparameters (like learning rate) on training stability.
[Uploading DQN.py…import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Q_network(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Q_network,self).__init__()
        self.fc1 = nn.Linear(state_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    def forward(self, state):
        x = F.relu(self.fc1(state))
        y = F.relu(self.fc2(x))
        return self.fc3(y)
class replay_buffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long()
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float()
        return (states,actions, rewards, next_states, dones)
    def __len__(self):
        return len(self.memory)
class DQN_agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = 64
        self.buffer_size = 10000
        self.update_every = 4
        self.lr = 0.001
        self.gamma = 0.99
        self.tau = 1e-3

        #Q network and target network
        self.q_network_local = Q_network(state_size, action_size)
        self.q_network_target = Q_network(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network_local.parameters(),lr=self.lr)
        
        # experience replay
        self.memory = replay_buffer(self.buffer_size,self.batch_size)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):

        #add experiences into buffer
        self.memory.add(state,action,reward,next_state,done)

        # learning after some steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)
    def act(self, state, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.q_network_local.eval()
        with torch.no_grad():
            action_values = self.q_network_local(state)
        self.q_network_local.train()
        if random.random()>eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    def learn(self,experiences):
        states,actions,rewards,next_states, dones = experiences

        #get current Q value
        Q_expected = self.q_network_local(states).gather(1,actions)

        # calculate Q value
        Q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1-dones))
        
        # calculate loss
        loss = F.mse_loss(Q_expected,Q_targets)

        # minimum loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self.soft_update(self.q_network_local, self.q_network_target)

    def soft_update(self, local_model,target_model):
        for target_param, local_param in zip(target_model.parameters(),local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
    
    # train dqn
def train_dqn(agent,env,n_episodes=3000, max_t=1000, eps_start=1.0,eps_end=0.01,eps_decay=0.995):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    for i_episode in range(1,n_episodes):
        state,info = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state,eps)
            next_state,reward,terminated, truncated,info = env.step(action)
            done = terminated or truncated
            agent.step(state,action,reward,next_state,done)
            state = next_state
            score+=reward
            if done:
                break

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end,eps_decay*eps)
        avg_score = np.mean(scores_window)
        if i_episode % 100 == 0:
            print(f'Episode {i_episode}\tAverage Score: {avg_score:.2f}')
        if avg_score >= 200.0:
            print(f'Environment solved in {i_episode-100} episodes!\tAverage Score: {avg_score:.2f}')
            torch.save(agent.q_network_local.state_dict(),'checkpoint.pth')
            break
import gymnasium as gym
if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQN_agent(state_size,action_size)

    scores = train_dqn(agent,env)

    env.close()]()

## Results
After training for 500 episodes, the agent achieved an average reward of over 195 (out of a maximum of 200) in the CartPole environment.

![Training Progress](/reward_plot.png) # 这里显示你上传的图片

## How to Run the Code
1. Install the required packages: `pip install gym numpy torch`
2. Run the training script: `python dqn.py`

## About Me
I am a graduate student in Computer Science transitioning from a management background. I am passionate about reinforcement learning and explainable AI.
