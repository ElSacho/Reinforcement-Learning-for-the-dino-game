import torch
import random
import numpy as np
from collections import deque
from dinoEnv import DinoGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY) #popleft if on depasse
        self.model = Linear_QNet(4, 48,4)
        self.trainer = QTrainer(self.model,lr=LR, gamma=self.gamma) 
    
    def get_state(self, game):
        # distance to next obstacle, height of the obstacle (2 ou beaucoup), playerY position, gap beetween obstacle, 
        # biais en 2e temps
        if len(game.obstacles.obstacles)>0:
            distance_to_next_obstacle = game.obstacles.obstacles[0].obs[0].x-game.player.x
            height_of_the_obstacle = game.obstacles.obstacles[0].type
        else:
            distance_to_next_obstacle = game.w
            height_of_the_obstacle = 0
        playerY_position = game.player.y
            
        if len(game.obstacles.obstacles)>1:
            gap_between_obstacles=game.obstacles.obstacles[1].obs[0].x-game.obstacles.obstacles[0].obs[0].x
        else: 
            gap_between_obstacles = game.w
            
        state = [
            distance_to_next_obstacle,
            height_of_the_obstacle,
            playerY_position,
            gap_between_obstacles,
        ]
        
        return state
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #Liste of tuples
        else:
            mini_sample =self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)  
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        # random moves: tradeoff exploration/exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,3)
            final_move[move]=1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move
    
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = DinoGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)
        
        # get move
        final_move = agent.get_action(state_old)
        
        # perform move and get new state
        reward, done, score  = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # train short memory
        agent.train_short_memory(state_new, final_move, reward, state_new, done)
        
        #remember
        agent.remember(state_new, final_move, reward, state_new, done)
        
        if done:
            #train long memory, plot result
            game.reset()
            agent.n_games+=1
            agent.train_long_memory()
            if score> record:
                record = score
                agent.model.save()
                
            print('Game', agent.n_games, 'Score', score, 'Record', record)
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()