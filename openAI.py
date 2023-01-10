import random
import numpy as np
import dinoEnv
import dinoEnvWithoutDisplay

# Fonction pour mettre à jour la politique en utilisant la cross-entropie
def mettre_a_jour_politique_cross_entropy(politique, cible, alpha):
  divergence = np.sum(np.abs(cible - politique) * cible, axis=1)
  moyenne_divergence = np.mean(divergence)
  politique += alpha * (cible - politique) / moyenne_divergence

# Fonction pour mettre à jour la table Q en utilisant le Q-learning
def mettre_a_jour_q(q, etat, action, prochain_etat, recompense, gamma, alpha):
  # Calcul de la valeur Q attendue pour l'état et l'action choisis
  q_attendu = recompense + gamma * max(q[prochain_etat[0]][prochain_etat[1]])
  
  # Mise à jour de la valeur Q pour l'état et l'action choisis
  q[etat[0]][etat[1]][action] = (1 - alpha) * q[etat[0]][etat[1]][action] + alpha * q_attendu

# Fonction principale de RL
def apprendre(env, nb_episodes, alpha, gamma):
  # Initialisation de la table Q et de la politique
  q = np.zeros((5, 5, 4))
  politique = np.ones((5, 5, 4)) / 4
  
  # Boucle d'apprentissage
  for i in range(nb_episodes):
    # Réinitialisation de l'environnement
    etat = env.reset()
    
    # Boucle d'interaction avec l'environnement
    fini = False
    while not fini:
        # Sélection d'une action en utilisant la politique et la table Q
        print(politique)
        print(etat)
        action = np.random.choice(4, p=politique[etat[0]][etat[1]])
      
        # Interaction avec l'environnement
        prochain_etat, recompense, fini = env.step(action)

        # Mise à jour de la table Q en utilisant le Q-learning
        mettre_a_jour_q(q, etat, action, prochain_etat, recompense, gamma, alpha)

        # Mise à jour de la politique en utilisant la cross-entropie
        cible = np.exp(q[etat[0]][etat[1]]) / np.sum(np.exp(q[etat[0]][etat[1]]))
        mettre_a_jour_politique_cross_entropy(politique[etat[0]][etat[1]], cible, alpha)

        # Passage à l'état suivant
        etat = prochain_etat
    print(recompense)
        
env = dinoEnvWithoutDisplay.dinoEnvWithoutDisplay()
nb_episodes = 10
alpha = 0.1
gamma = 0.9
apprendre(env, nb_episodes, alpha, gamma)