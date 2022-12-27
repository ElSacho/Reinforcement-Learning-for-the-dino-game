from cgitb import reset
from os import remove
from turtle import width
import pygame
import random
from enum import Enum
from collections import namedtuple

Point = namedtuple('Point', 'x, y')
# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 25


class Obstacles:
    def __init__(self, width, bottumMargin):
        self.obstacles = []
        self.w=width
        self.bottumMargin=bottumMargin
        self.t=0
    
    def generateObstacle(self, speed):
        if self.t>0:
            self.t-=1
        else:
            x = random.randint(0, 3)
            if x ==1:
                self.t=25*10/speed
                y = random.randint(0, 3)
                if y==1:
                    self.obstacles.append(Obstacle(self.w, self.bottumMargin, 2))
                else :
                    self.obstacles.append(Obstacle(self.w, self.bottumMargin, 1))
        
    def _moveObstacles(self, speed):
        for obstacle in self.obstacles:
            obstacle._moveObstacle(speed)
            if obstacle.toKill==True:
                self.obstacles.remove(obstacle)
            
    def drawObstacles(self, game):
        for obstacle in self.obstacles:
            obstacle.drawObstacle(game)
    
        
class Obstacle:
    def __init__(self, width, bottumMargin, type):
        self.type=type
        if type ==1:
            self.obs = [Point(width,bottumMargin-j*BLOCK_SIZE) for j in range(2)]
        if type == 2:
            self.obs = [Point(width,bottumMargin-j*BLOCK_SIZE) for j in range(2,50)]
        self.toKill=False

    def _moveObstacle(self, speed):
        NewObstacle = [Point(pt.x-speed, pt.y) for pt in self.obs]
        if self.obs[0].x<BLOCK_SIZE:
            self.toKill=True
            self.obs=[]
            return
        self.obs = NewObstacle

    def drawObstacle(self, game):
        for pnt in self.obs:
            pygame.draw.rect(game.display, RED, pygame.Rect(pnt.x, pnt.y, BLOCK_SIZE, BLOCK_SIZE))