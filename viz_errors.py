import torch
import numpy as np
import matplotlib.pyplot as plt
from models import MLP

# Fonction pour charger le modèle
def load_model(checkpoint_path, input_size, hidden_size, output_size):
    model = MLP(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

# Charger les données
X = np.load('./data/X.npy')  # Données d'observations
y = np.load('./data/y.npy')  # Données de vérité terrain (trajectoires réelles)

# Paramètres
n_robots = 1  # Nombre de robots à afficher
checkpoint_path = './models/128_neur/model_epoch_40.pth'  # Chemin du checkpoint

# Paramètres du modèle
input_size = X.shape[1]
hidden_size = 128
output_size = y.shape[1]

# Charger le modèle
model = load_model(checkpoint_path, input_size, hidden_size, output_size)

# Sélectionner aléatoirement 5 robots parmi les données
indices = np.random.choice(X.shape[0], size=n_robots, replace=False)
X_sample = X[indices]
y_sample = y[indices]

# Prédictions du modèle
with torch.no_grad():
    X_tensor = torch.tensor(X_sample, dtype=torch.float32)
    y_pred = model(X_tensor).numpy()

# Plot des trajectoires
plt.figure(figsize=(10, 6))

for i in range(n_robots):
    # Vérité terrain (y contient des [x1, y1, x2, y2, ...])
    x_real = y_sample[i, ::2]  # Prendre les indices pairs pour les x
    y_real = y_sample[i, 1::2]  # Prendre les indices impairs pour les y

    # Prédictions (y_pred contient des [x1, y1, x2, y2, ...])
    x_pred = y_pred[i, ::2]  # Indices pairs pour les x prédites
    y_pred_i = y_pred[i, 1::2]  # Indices impairs pour les y prédites

    # Plotter la vérité terrain et la prédiction
    plt.plot(x_real, y_real, 'go-', label=f'Robot {i+1} Vérité Terrain' if i == 0 else "", markersize=5)
    plt.plot(x_pred, y_pred_i, 'ro--', label=f'Robot {i+1} Prédiction' if i == 0 else "", markersize=5)

# Configuration du plot
plt.xlabel('Position X')
plt.ylabel('Position Y')
plt.title(f'Vérité terrain vs Prédiction du modèle pour {n_robots} robots')
plt.legend()
plt.grid(True)
plt.show()
