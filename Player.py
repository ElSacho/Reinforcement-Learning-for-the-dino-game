from cgitb import reset
from os import remove
from turtle import width
import pygame
import random
from enum import Enum
from collections import namedtuple
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


class Player:
    
    def __init__(self, leftMargin, bottumMargin):
        self.x = leftMargin+4*BLOCK_SIZE
        self.y = bottumMargin
        self.sol = bottumMargin
        self.isBaissing = False
        self.isSauting = False
        self.taille=0
        self.t=0
        self.pt = [Point(self.x,self.y-j*BLOCK_SIZE) for j in range(3)]
    
    
    def _movePlayer(self, action):
        # action = [petit Saut, grand saut, se baisser, ne rien faire]
        if self.isSauting == False:
            if np.array_equal(action, [1,0,0,0]):
                self.isSauting=True
                self.taille=1
            if np.array_equal(action, [0,1,0,0]):
                self.isSauting=True
                self.taille=2
            if np.array_equal(action, [0,0,1,0]):
                if self.taille not in [1,2]:
                    self.taille=-1

        if self.isBaissing and self.taille!=-1:
            self._goUp()
        if self.isSauting:
            self.saut(self.taille)
        if self.isSauting == False and self.taille==-1:
            self._goDown()
            
        
    
    def _goDown(self):
        self.pt = [Point(self.x,self.y) ]
        self.isBaissing=True
    
    def _goUp(self):
        self.pt = [Point(self.x,self.y-j*BLOCK_SIZE) for j in range(3)]
        self.isBaissing=False
        
    def drawPlayer(self, game):
        for pnt in self.pt:
            pygame.draw.rect(game.display, BLACK, pygame.Rect(pnt.x, pnt.y, BLOCK_SIZE, BLOCK_SIZE))

    def saut(self, taille):
        if taille==0:
            return
        gravity = 500
        self.t+=0.3
        if taille==1:
            pos = -gravity*(-self.t*self.t/2+self.t)
        else:
            pos = -gravity*(-(self.t)*(self.t-4))/8
        if pos>0:
            self.y = self.sol
            self.taille=0
            self.isSauting=False
            self.t=0
        else:
            self.y=pos+self.sol
        self.pt = [Point(self.x,self.y-j*BLOCK_SIZE) for j in range(3)]

    def isCollision(self, obstacles, game):
        gameOver=False
        for obstacle in obstacles.obstacles:
            if obstacle.obs[0].x > self.x+BLOCK_SIZE:
                break
            # Si l'obstacle est à la meme abscisse que notre player, on compare les ordonnées 
            if obstacle.obs[0].x + BLOCK_SIZE >= self.x:
                if obstacle.type==1:
                    if obstacle.obs[0].y-3*BLOCK_SIZE < self.y:
                        gameOver=True
                        return gameOver
                if obstacle.type==2:
                    if not self.isBaissing:
                        return True
            if obstacle.obs[0].x + BLOCK_SIZE < self.x:
                game.score+=1
                print(game.score)
                obstacles.obstacles.remove(obstacle)

            return False