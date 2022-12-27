from cgitb import reset
from os import remove
from turtle import width
import pygame
import random
from enum import Enum
from collections import namedtuple
import Obstacles
import Player
import numpy as np


Point = namedtuple('Point', 'x, y')
# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 25

class dinoEnvWithoutDisplay:
    def __init__(self, width=680, height=420, leftMargin=20, bottumMargin=400, maxSpeed=50):
        self.w = width
        self.h = height
        self.leftMargin=leftMargin
        self.bottumMargin=bottumMargin
        self.maxSpeed = maxSpeed
        self.observation_space = 5
        self.action_space = 4
        self.resetIni()
        
    def resetIni(self):
        self.sol=[Point(k*BLOCK_SIZE, self.bottumMargin+BLOCK_SIZE) for k in range(self.w//BLOCK_SIZE)]
        self.obstacles = Obstacles.Obstacles(self.w, self.bottumMargin)
        self.player = Player.Player(self.leftMargin, self.bottumMargin)
        self.score=0
        self.vitesse=5
        
    def reset(self):
        self.sol=[Point(k*BLOCK_SIZE, self.bottumMargin+BLOCK_SIZE) for k in range(self.w//BLOCK_SIZE)]
        self.obstacles = Obstacles.Obstacles(self.w, self.bottumMargin)
        self.player = Player.Player(self.leftMargin, self.bottumMargin)
        self.score=0
        self.vitesse=5
        return self.get_obs()
    
    def updateVitesse(self):
        if self.vitesse < self.maxSpeed :
            self.vitesse+=0.01

    def play_step(self, action):
         # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.player._movePlayer(action) 
        self.obstacles._moveObstacles(self.vitesse)
        gameOver = self.player.isCollision(self.obstacles, self)
        self.obstacles.generateObstacle(self.vitesse)
        self.updateVitesse()  
        # self.clock.tick(SPEED)
        reward = 1
        if gameOver:
            reward = -10
        obs = np.array(self.get_obs())
        
        return obs, reward, gameOver    
    
    def step(self, action):
        return self.play_step(action)
    
    def get_obs(self):
        speed = self.vitesse / self.maxSpeed
        if len(self.obstacles.obstacles) == 0:
            Dmur = 1
            Dpont = 1
            Gap = 1
        elif len(self.obstacles.obstacles) == 1:
            obstacle = self.obstacles.obstacles[0]
            Gap = 1
            if obstacle.type == 1:
                Dmur = obstacle.obs[0].x / self.w
                Dpont = 1
            else : 
                Dmur = 1
                Dpont = obstacle.obs[0].x / self.w
            Gap = 1
        else :
            obstacle1 = self.obstacles.obstacles[0]
            obstacle2 = self.obstacles.obstacles[1]
            if obstacle1.type == 1 and obstacle2.type == 1:
                Dmur = obstacle1.obs[0].x / self.w
                Dpont = 1
            elif obstacle1.type == 2 and obstacle2.type == 2:
                Dmur = 1
                Dpont = obstacle1.obs[0].x / self.w
            elif obstacle1.type == 1 and obstacle2.type == 2:
                Dmur = obstacle1.obs[0].x / self.w
                Dpont = obstacle2.obs[0].x / self.w
            else :
                Dmur = obstacle2.obs[0].x / self.w
                Dpont = obstacle1.obs[0].x / self.w
            Gap = ( obstacle2.obs[0].x - obstacle1.obs[0].x ) / self.w
        Ypos = self.player.y
        return [speed, Ypos, Gap, Dmur, Dpont]

