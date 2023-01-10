from dinoEnv import dinoEnv
from neuronalNetwork import NeuronalNetwork
import Player



class Agents:
    def __init__(self, nbrAgents, layers):
        self.game = dinoEnv()
        self.nbrArgents = nbrAgents
        self.agents = []
        for i in range(nbrAgents):
            network = NeuronalNetwork(layers)
            self.agents.append(Agent(network))
            
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
        self.game=dinoEnv()
            
    def play_step(self):
        for agent in self.agents:
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
    
class Agent:
    def __init__(self, network, leftMargin=20, bottumMargin=400):
        self.network = network
        self.isDead = False
        self.score = 0
        self.player = Player.Player(leftMargin, bottumMargin)

        
    def play_step(self, env):
        if self.isDead:
            return
        obs = env.get_obs()
        action = self.network.feedForward(obs)
        gameOver = self.game.play_step(action)
        self.score += 1
        self.isDead = gameOver
    
    def get_action(self, obs):
        return self.network.feedForward(obs)
    

    
    