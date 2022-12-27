from cgitb import reset
from os import remove
from turtle import width
import pygame
import random
from enum import Enum
from collections import namedtuple
import Obstacles
import Player

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

    
Point = namedtuple('Point', 'x, y')
# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 25



class dinoEnv:
    def __init__(self, width=680, height=420, leftMargin=20, bottumMargin=400):
        self.w = width
        self.h = height
        self.leftMargin=leftMargin
        self.bottumMargin=bottumMargin
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Dino')
        self.clock = pygame.time.Clock()
        self.reset()
        
    def reset(self):
        self.sol=[Point(k*BLOCK_SIZE, self.bottumMargin+BLOCK_SIZE) for k in range(self.w//BLOCK_SIZE)]
        self.obstacles = Obstacles.Obstacles(self.w, self.bottumMargin)
        self.player = Player.Player(self.leftMargin, self.bottumMargin)
        self.score=0
        self.vitesse=5
    
    def updateVitesse(self):
        if self.vitesse < 50:
            self.vitesse+=0.01
    
    def drawSol(self):
        for pt in self.sol:
            pygame.draw.rect(self.display, BLACK, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        
        
    def _update_ui(self):
        self.display.fill(WHITE)
        
        self.drawSol()
        self.obstacles.drawObstacles(self)
        self.player.drawPlayer(self)

        
        text = font.render("Score: " + str(self.score), True, BLACK)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
    
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
                  
        self._update_ui()
        self.clock.tick(SPEED)
        reward = 2
        if gameOver:
            reward=-10
        
        return reward, gameOver, self.score
    
    def step(self, action):
        pass

