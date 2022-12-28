import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
import gym
import dinoEnv
import dinoEnvWithoutDisplay

# RL Gym
env = dinoEnvWithoutDisplay.dinoEnvWithoutDisplay()

# Initialisation
n = 100  # number of candidate policies
top_k = 0.10  # top % selected for next iteration
mean = np.zeros((5,4))  # shape = (n_parameters, n_actions)
stddev = np.ones((5,4))  # shape = (n_parameters, n_actions)

def get_batch_weights(mean, stddev, n):
    mvn = tfd.MultivariateNormalDiag(
        loc=mean,
        scale_diag=stddev)
    return mvn.sample(n).numpy()

def policy(obs, weights):
    obs = np.array(obs)
    weights = np.array(weights)
    return np.argmax(obs @ weights + weights[4])

def run_trial(weights, render=False):
    obs = env.reset()
    done = False
    reward = 0
    while not done:
        a = policy(obs, weights)
        obs, r, done = env.step(a)
        reward += r
        if render:
            pass
            # env.render()
    return reward

def get_new_mean_stddev(rewards, batch_weights):
    idx = np.argsort(rewards)[::-1][:int(n*top_k)]
    if rewards[idx[0]] > 30:
        mean = np.mean(batch_weights[[0]], axis =0)
        stddev = np.sqrt(np.var(batch_weights[idx[0]], axis=0))
    else :
        mean = np.mean(batch_weights[idx], axis=0)
        stddev = np.sqrt(np.var(batch_weights[idx], axis=0))
    return mean, stddev

for i in range(10000):
    batch_weights = get_batch_weights(mean, stddev, n)
    rewards = [run_trial(weights) for weights in batch_weights]
    mean, stddev = get_new_mean_stddev(rewards, batch_weights)
    rewards.sort(reverse=True)
    print(rewards[:20])