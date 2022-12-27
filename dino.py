from os import remove
from turtle import width
import pygame
import random
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')
# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 25



class DinoGame:
    def __init__(self, width=680, height=420, leftMargin=20, bottumMargin=400):
        self.w = width
        self.h = height
        self.score=0
        self.vitesse=5
        self.leftMargin=leftMargin
        self.bottumMargin=bottumMargin
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Dino')
        self.clock = pygame.time.Clock()
        self.sol=[Point(k*BLOCK_SIZE, bottumMargin+BLOCK_SIZE) for k in range(width//BLOCK_SIZE)]
        self.obstacles = Obstacles(width, bottumMargin)
        self.player = Player(leftMargin, bottumMargin)
    
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
    
    def play_step(self):
         # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if self.player.isSauting == False:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.player.taille=1
                        self.player.isSauting = True
                    if event.key == pygame.K_RIGHT:
                        self.player.taille=2
                        self.player.isSauting = True
                    if event.key == pygame.K_DOWN:
                        if self.player.taille not in [1,2]:
                            self.player.taille=-1
        

        self.player._movePlayer() 
        self.obstacles._moveObstacles(self.vitesse)
        gameOver = self.player._isCollision(self.obstacles, self)
        self.obstacles.generateObstacle(self.vitesse)
        self.updateVitesse()
                  
        self._update_ui()
        self.clock.tick(SPEED)
        return gameOver


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

    
class Player:
    def _movePlayer(self):
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
    
    def __init__(self, leftMargin, bottumMargin):
        self.x = leftMargin+4*BLOCK_SIZE
        self.y = bottumMargin
        self.sol = bottumMargin
        self.isBaissing = False
        self.isSauting = False
        self.taille=0
        self.t=0
        self.pt = [Point(self.x,self.y-j*BLOCK_SIZE) for j in range(3)]
        
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

    def _isCollision(self, obstacles, game):
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
                obstacles.obstacles.remove(obstacle)

            return False
                    
    
                
        
if __name__ == '__main__':
    game = DinoGame()
    
    #game loop
    while True:
        game_over = game.play_step()
    #    
        if game_over == True:
            break     
    pygame.quit()