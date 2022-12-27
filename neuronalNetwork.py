import random
import math

class NeuronalNetwork():
    
    def __init__(self, layers):
        self.layers = layers # [3; 4; 5 ;3] -> 3 entrees ; 4 en hidden 1, 5 en hidden 2, 3 sorties
        self.initNeurons() #[[0,1,2][0,1,2,3][5][3]]
        self.initWeight() #[[PoidsDuNeurone1 aux neuronnes de la couche precedente[w1, w2, w3], [x1,x2,x3]]]
       
        
    # initialiser un reseau à partir d'un poids donné (faire une copie du neuronne)  
    def copy(self, weights):
        layers = self.initLayers()
        copy = NeuronalNetwork(layers)
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    copy.weights[i][j][k]=weights[i][j][k]
        return copy
    
    def copy2(self, weights):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k]=weights[i][j][k]
        
    def initLayers(self):
        #pour chaque couche
        layers=[len(self.weights[0][0])]
        for i in range(len(self.weights)):
            layers.append(len(self.weights[i]))
        return layers
    
    def initNeurons(self):
        neuronList = []
        for i in range(len(self.layers)):
            listCouche=[]
            for j in range(self.layers[i]):
                listCouche.append(j)
            neuronList.append(listCouche)
        self.neurons=neuronList
            
    def initWeight(self):
        weightList=[]
        for i in range(1, len(self.layers)):
            layerWeightList=[]
            neuronInPreviousLayer= self.layers[0]
            # pour chaque couche
            for j in range(self.layers[i]):
                neuronsWeight=[]
                #entre les reseaux du neuronne precedent et le neuronne actuel on genere aleatoirement des valeurs
                for k in range(neuronInPreviousLayer):
                    neuronsWeight.append(random.uniform(-1,1))
                #chaque couche a ses valeurs
                layerWeightList.append(neuronsWeight)
            #weightList[couche][neuronne][weight]
            weightList.append(layerWeightList)
        self.weights=weightList
        
    def addFitness(self, fit):
        self.fitness += fit
    
    def setFitness(self, fit):
        self.fitness = fit
    
    def getFitness(self):
        return self.fitness   
        
    def compareTo(self, other):
      #  if other == NULL:
      #      return 1
        if self.fitness> other.fitness:
            return 1
        if self.fitness < other.fitness:
            return -1
        return 0
    
    def feedForward(self, inputs):
        
        # ajouter les inputs dans la matrice de neuronnes
        for i in range(len(inputs)):
            self.neurons[0][i]=inputs[i]
    
        #Calculs
        #Pour chaque couche à partir de la seconde couche
        for i in range(1, len(self.layers)):
            #Pour chaque neuron
            for j in range(len(self.neurons[i])):
                val = 0
                for k in range(len(self.neurons[i-1])):
                    val += self.weights[i-1][j][k]*self.neurons[i-1][k] # Calcul de la valeur de chaque neurone
                self.neurons[i][j] = math.tanh(val) # fonction d'activation
        return self.neurons[len(self.neurons)-1] #renvoie l'output
            
    def mutation(self, tauxMutation):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    weight=self.weights[i][j][k]
                    randNum = random.uniform(0,1)
                    if (randNum<tauxMutation):
                        newWeight=random.uniform(-1,1)
                        weight = newWeight
                    self.weights[i][j][k] = weight
        
        
    def get_mutation(self, tauxMutation):
        mut=self.weights
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    weight=self.weights[i][j][k]
                    randNum = random.uniform(0,1)
                    if (randNum<tauxMutation):
                        newWeight=random.uniform(-1,1)
                        weight = newWeight
                    mut[i][j][k] = weight
        return mut
        
    
