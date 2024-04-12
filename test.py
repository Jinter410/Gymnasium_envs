# Correction et vectorisation complète du calcul des distances d'intersection sans instructions conditionnelles
import numpy as np
# Générer 3 angles aléatoires en degrés
angles = np.random.randint(0, 360, 3)

# Conversion des angles en radians pour le calcul
angles_radians = np.radians(angles)

# Calculer cosinus et sinus pour éviter la division par zéro
cos_angles = np.cos(angles_radians)
sin_angles = np.sin(angles_radians)

# Dimensions du rectangle divisées par deux (pour calculer à partir du centre)
demi_largeur = 0.8
demi_hauteur = 0.3
rectangle_shape = [demi_largeur, demi_hauteur]
print("a:",[cos_angles, sin_angles], "b:",np.stack([cos_angles, sin_angles], axis=-1))
distances_t = np.min(rectangle_shape / np.abs(np.stack([cos_angles, sin_angles], axis=-1)), axis=-1)

# Calcul des distances d'intersection sans utiliser de conditions explicites
# Utiliser np.clip pour éviter la division par zéro en remplaçant 0 par un très petit nombre
distances_x = demi_largeur / np.clip(np.abs(cos_angles), 1e-10, None)
distances_y = demi_hauteur / np.clip(np.abs(sin_angles), 1e-10, None)

# Le minimum des distances sur les axes x et y est la distance d'intersection
distances = np.minimum(distances_x, distances_y)
print(distances_t, distances)

angles, distances.tolist()
# print(angles, distances.tolist())

# Display rectangle
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect(1.0)
rectangle = plt.Rectangle((-demi_largeur, -demi_hauteur), 2*demi_largeur, 2*demi_hauteur, fill=False)
ax.add_artist(rectangle)
# Plot intersection points
for angle, distance in zip(angles, distances):
    x = distance * np.cos(np.radians(angle))
    y = distance * np.sin(np.radians(angle))
    ax.plot([0, x], [0, y], 'o-')
plt.show()
