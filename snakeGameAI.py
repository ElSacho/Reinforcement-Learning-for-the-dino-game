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
SPEED = 20

class SnakeGame:

    def __init__(self, width=640, height=480):
        self.w=width
        self.h=height
        # Prepare for display        
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

        
    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.y-2*BLOCK_SIZE, self.head.y)]
        self.score=0
        self.food=False
        self._placeFood()
        
        
    def play_step(self):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
        # Move
        self._move(self.direction)
        self.snake.insert(0, self.head)
        
        # Check if the game is over
        gameOver = False
        if self._isCollision():
            gameOver=True
            reward = -10
            return reward, gameOver, self.score
        
        # check if on est sur du food
        if self.head == self.food:
            self.score+=1
            self._placeFood()
        else:
            self.snake.pop()
            
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        reward = 2
        # 6. return game over and score
        return reward, gameOver, self.score
    
    def _placeFood(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food=Point(x,y)
        if self.food in self.snake:
            self._placeFood()
    
    def _isCollision(self):
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True
        if self.head in self.snake[1:]:
            return True
        return False 
    
    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        if direction == Direction.LEFT:
            x -= BLOCK_SIZE
        if direction == Direction.UP:
            y -= BLOCK_SIZE
        if direction == Direction.DOWN:
            y += BLOCK_SIZE
        self.head = Point(x,y)
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for boutSnake in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(boutSnake.x, boutSnake.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(boutSnake.x+4, boutSnake.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    
    
if __name__ == '__main__':
    game = SnakeGame()
    
    #game loop
    while True:
        game_over, score = game.play_step()
    #    
        if game_over == True:
            break
        
    print('Final Score', score)        
    pygame.quit()
