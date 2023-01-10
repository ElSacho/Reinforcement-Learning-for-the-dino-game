from cgitb import reset
from os import remove
from turtle import width
import pygame
import random
from enum import Enum
from collections import namedtuple
import Obstacles
import Player
import agents

pygame.init()
font = pygame.font.Font('Reinforcement-Learning-for-the-dino-game/arial.ttf', 25)


    
Point = namedtuple('Point', 'x, y')
# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 25

class DinoGameAI:
    def __init__(self, nbrAgents, layers= [5, 3,3,4], width=680, height=420, leftMargin=20, bottumMargin=400):
        self.w = width
        self.h = height
        self.leftMargin=leftMargin
        self.bottumMargin=bottumMargin
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Dino')
        self.clock = pygame.time.Clock()
        self.agents = agents.Agents(nbrAgents, layers)
        self.maxSpeed = 50
        self.reset()
        
    def reset(self):
        self.sol=[Point(k*BLOCK_SIZE, self.bottumMargin+BLOCK_SIZE) for k in range(self.w//BLOCK_SIZE)]
        self.obstacles = Obstacles.Obstacles(self.w, self.bottumMargin)
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
        for agent in self.agents.agents:
            if not agent.isDead :
               agent.player.drawPlayer(self)
            
        text = font.render("Score: " + str(self.score), True, BLACK)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
    
    def play_step(self):
         # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                
        for agent in self.agents.agents:
            if not agent.isDead :
                obs = self.get_obs()
                obs.append(agent.player.y / self.h)
                action = agent.get_action(obs)
                agent.player._movePlayer(action)
                agent.isDead = agent.player.isCollision(self.obstacles, self)
 
        self.obstacles._moveObstacles(self.vitesse)
        self.obstacles.generateObstacle(self.vitesse)
        self.updateVitesse()
                  
        self._update_ui()
        self.clock.tick(SPEED)
        return self.score

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
            return [speed, Gap, Dmur, Dpont]

if __name__ == '__main__':
    game = DinoGameAI(100)
    
    #game loop
    while True:
        game.play_step()
        game.agents.sortAgents()
        if game.agents.agents[0].isDead:
            game.agents.updateAgents()
       
    pygame.quit()