import gym
from collections import namedtuple
import matplotlib.pyplot as plt 
from skimage import io
import cv2
import numpy as np
from tensorboardX import SummaryWriter
import dinoEnv
import dinoEnvWithoutDisplay

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


HIDDEN_SIZE = 128
BATCH_SIZE = 100
PERCENTILE = 70

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 56),
            nn.ReLU(),
            nn.Linear(56, n_actions)
        )

    def forward(self, x):
        return self.net(x)
    
    def mutate(self, rate):
        # Transformation aléatoire de 10% des poids du modèle
        for p in self.parameters():
            mask = torch.zeros_like(p, dtype=torch.bool)
            n = p.numel()
            mask[:int(n * 0.1)] = 1  # sélectionne les 10% premiers éléments
            mask = mask[torch.randperm(n)]  # mélange les éléments sélectionnés
            p.data[mask] = nn.init.uniform_(p.data[mask], -1, 1)  # transforme les éléments sélectionnés de manière aléatoire
    
class Agent():
    def __init__(self, obs_size, hidden_size, n_actions):
        self.net = Net(obs_size, HIDDEN_SIZE, n_actions)
        self.reward = 0
    
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action']) 
    
def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=-1)
    while True:
        obs_v = torch.FloatTensor(obs)
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()
        action = np.random.choice(len(act_probs), p=act_probs)
        # print(f'action : {action}')
        next_obs, reward, is_done = env.step(action)
        episode_reward += reward
        # print(f'episode_reward : {episode_reward}')
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)
        if is_done:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            # print(episode_reward)
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs
        
def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, steps))
        train_act.extend(map(lambda step: step.action, steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean

if __name__ == "__main__":
    # env = dinoEnv.dinoEnv()
    env = dinoEnvWithoutDisplay.dinoEnvWithoutDisplay()
    obs_size = env.observation_space
    n_actions = env.action_space
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-dinosaure")

    for iter_no, batch in enumerate(iterate_batches(
            env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = \
            filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 1000:
            print("Solved!")
            break
    writer.close()