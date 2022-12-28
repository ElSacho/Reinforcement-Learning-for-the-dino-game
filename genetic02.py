import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
import gym
import dinoEnv
import dinoEnvWithoutDisplay
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        self.obs_size = obs_size
        self.hidden_size = hidden_size
        self.n_actions = n_actions
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)
    
    def mutate(self, percentage):
        n_weights = sum(p.numel() for p in self.parameters())  # nombre total de poids dans le modèle
        indices = torch.randperm(n_weights)[:int(n_weights * percentage)]  # sélection de 10% des indices aléatoirement
        # Modification des poids sélectionnés
        for i, p in enumerate(self.parameters()):
            # Flattening du tenseur des poids
            flat_weights = p.view(-1)
            if i == 0:
                concate = flat_weights
            else :
                concate = torch.cat((concate, flat_weights), dim=0)
        concate[indices] = 2*random.random()-1
        weights = self.state_dict()

        # Modification des poids
        weights['net.0.weight'].data = concate[:self.hidden_size*self.obs_size].view(self.hidden_size, self.obs_size)
        weights['net.0.bias'].data = concate[:self.hidden_size]
        weights['net.1.weight'].data = concate[self.hidden_size*self.obs_size:self.hidden_size*self.obs_size+self.hidden_size*self.n_actions].view(self.n_actions, self.hidden_size)
        weights['net.1.bias'].data = concate[-self.n_actions:]

        # Mise à jour du modèle avec les nouveaux poids
        self.load_state_dict(weights)

class Agent():
    def __init__(self, obs_size, hidden_size, n_actions):
        self.net = Net(obs_size, hidden_size, n_actions)
        self.reward = 0
        
    def playAgent(self, env):
        self.reward = 0.0
        obs = env.reset()
        sm = nn.Softmax(dim=-1)
        while True:
            obs_v = torch.FloatTensor(obs)
            act_probs_v = sm(self.net(obs_v))
            act_probs = act_probs_v.data.numpy()
            action = np.random.choice(len(act_probs), p=act_probs)
            next_obs, reward, is_done = env.step(action)
            self.reward += reward
            obs = next_obs
            if is_done:
                return
        

class Agents():
    def __init__(self, obs_size, hidden_size, n_actions, n_agent):
        self.agents=[]
        self.rewards = np.zeros(n_agent)
        for _ in range(n_agent):
            self.agents.append(Agent(obs_size, hidden_size, n_actions))
            
    def playAllAgents(self, env):
        for i in range(len(self.agents)):
            self.agents[i].playAgent(env)
            self.rewards[i] = self.agents[i].reward
            self.agents[i].reward = 0
    
    def updateAgents(self):
        idx = np.argsort(self.rewards)[::-1]
        tot = len(idx)
        for i in range(tot):
            if i < tot/4:
                continue
            elif i < tot/2:
                self.agents[i].net.mutate(0.01)
            elif i < 3*tot/4:
                self.agents[i].net.mutate(0.05)
            else :
                self.agents[i].net.mutate(0.09)
                
# RL Gym
obs_size = 5
hidden_size = 128
n_actions = 4
n_agents = 10
# env = dinoEnvWithoutDisplay.dinoEnvWithoutDisplay()
env = dinoEnv.dinoEnv()
agents = Agents(obs_size, hidden_size, n_actions, n_agents)  

for i in range(100):
    agents.playAllAgents(env)
    agents.updateAgents()
    print(np.max(agents.rewards[i]))