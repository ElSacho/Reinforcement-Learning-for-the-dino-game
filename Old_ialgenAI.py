from game import world
from neuronalNetwork import NeuronalNetwork
import pygame




class Agents:
    def __init__(self, nbrAgents, layers):
        self.nbrArgents = nbrAgents
        self.bestScore=0
        self.nbrGeneration = 0
        self.agents = []
        for i in range(nbrAgents):
            network = NeuronalNetwork(layers)
            self.agents.append(Agent(network))
            
    def play_step(self):
        if self.agents[0].isDead:
            return True
        self.agents[0].play_and_draw(self.bestScore, self.nbrGeneration)
        for agent in self.agents[1:]:
            if not agent.isDead:
                agent.play_step()
        self.sortAgents()
        return False
            
    def updateAgents(self):
        l = len(self.agents)
        for i in range(l):
            if i<l/4:
                self.agents[i].isDead = False
            if l/4<=i<l/2:
                self.agents[i].network.mutation(0.005)
                self.agents[i].isDead = False
            if l/2<=i<3*l/4:
                self.agents[i].network.mutation(0.02)
                self.agents[i].isDead = False
            if 3*l/4<=i:
                self.agents[i].network.mutation(0.09)
                self.agents[i].isDead = False
            
    def sortAgents(self):
        newAgents=[]
        for agent in self.agents:
            if not agent.isDead:
                newAgents.append(agent)  
        for agent in self.agents:
            if agent.isDead:
                newAgents.append(agent)
        self.agents = newAgents
        
    def reset(self):
        score=self.agents[0].score
        if score > self.bestScore:
            self.bestScore=score
        self.nbrGeneration+=1
        self.updateAgents()
        for agent in self.agents:
            agent.score=0
            agent.game=world()
             
        
class Agent:
    def __init__(self, network):
        self.game = world()
        self.network = network
        self.score = 0
        self.isDead = False
        
    def play_and_draw(self, bestScore, generation):
        if self.isDead:
            return
        action = self.get_action()
        print(action)
        gameOver = self.game.play_step_and_draw(action, bestScore, generation)
        self.score +=1
        self.isDead = gameOver
        
    def play_step(self):
        if self.isDead:
            return
        action = self.get_action()
        gameOver = self.game.play_step(action)
        self.score += 1
        self.isDead = gameOver
    
    def get_state(self):
        vitesse0 = self.game.ball.vitesse[0] / 300
        vitesse1 = self.game.ball.vitesse[1] / 300
        posPlayer0 = self.game.player.pos[0] / 480
        posPlayer1 = self.game.player.pos[1] / 640
        posBall0 = self.game.ball.pos[0] / 480
        posBall1 = self.game.ball.pos[1] / 640
        
        state = [vitesse0,
                vitesse1,
                posPlayer0,
                posPlayer1,
                posBall0,
                posBall1]
    
        return state
    
    def get_action(self):
        action = self.network.feedForward(self.get_state())
        action[0] = 480*action[0] 
        action[1] = 640*action[1] 
        return action
    
    def get_mutationAgent(self, tauxDeMutation):
        return self.network.get_mutation(tauxDeMutation)
    

def train():
    layers = [6,4,4,2]
    agents = Agents(100, layers)
    while True:
        game_over = agents.play_step()
        if game_over == True:
            print("gameOver")
            agents.reset()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                #vectorWeight=game.agents.getWeights()
                #fichier = open("data.py", "w")
               # fichier.write("weight=")
                #fichier.write(str(vectorWeight))
               # fichier.close()
                pygame.quit()
                quit()
                

if __name__ == '__main__':
    train()