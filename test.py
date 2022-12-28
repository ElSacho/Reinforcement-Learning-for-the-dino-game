import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random 
import torch

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        self.obs_size = obs_size
        self.hidden_size = hidden_size
        self.n_actions = n_actions
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)
    
    def mutate(self, percentage):
        n_weights = sum(p.numel() for p in self.parameters())  # nombre total de poids dans le modèle
        print(n_weights)
        indices = torch.randperm(n_weights)[:int(n_weights * percentage)]  # sélection de 10% des indices aléatoirement
        print(indices)
        # Modification des poids sélectionnés
        for i, p in enumerate(self.parameters()):
            # Flattening du tenseur des poids
            flat_weights = p.view(-1)
            if i == 0:
                concate = flat_weights
            else :
                concate = torch.cat((concate, flat_weights), dim=0)
            # Sélection des poids à modifier
            # weights_to_modify = flat_weights[indices]
            # # Modification des poids sélectionnés
            # weights_to_modify = random.random()
        # concate[indices] = random.random()
        concate[indices] = 0
        
        weights = self.state_dict()

        # Modification des poids
        weights['net.0.weight'].data = concate[:self.hidden_size*self.obs_size].view(self.hidden_size, self.obs_size)
        weights['net.0.bias'].data = concate[:self.hidden_size]
        print(weights['net.1.weight'].data)
        print(concate[self.hidden_size*self.obs_size:self.hidden_size*self.obs_size+self.hidden_size*self.n_actions].view(self.n_actions, self.hidden_size))
        weights['net.1.weight'].data = concate[self.hidden_size*self.obs_size:self.hidden_size*self.obs_size+self.hidden_size*self.n_actions].view(self.n_actions, self.hidden_size)
        weights['net.1.bias'].data = concate[-self.n_actions:]

        # Mise à jour du modèle avec les nouveaux poids
        self.load_state_dict(weights)
    
    def mutate2(self, rate):
        # Transformation aléatoire de 10% des poids du modèle
        for p in self.parameters():
            print(p)
            
            # mask = torch.zeros_like(p, dtype=torch.bool)
            # n = p.numel()
            # print(n)
            # mask[:int(n * rate)] = 1  # sélectionne les 10% premiers éléments
            # print(mask)
            # print(torch.randperm(n))
            # mask = mask[torch.randperm(n)]  # mélange les éléments sélectionnés
            # print(mask)
            # p.data[mask] = nn.init.uniform_(p.data[mask], -1, 1)  # transforme les éléments sélectionnés de manière aléatoire
            
obs_size = 2
hidden_size = 3
n_actions = 2

net = Net(obs_size, hidden_size, n_actions)

for i, p in enumerate(net.parameters()):
    print(p)

net.mutate(0.4)

for i, p in enumerate(net.parameters()):
    print(p)