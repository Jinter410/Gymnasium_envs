import numpy as np
import matplotlib.pyplot as plt

# Paramètres
angles = np.random.randint(0, 360, 15)
angles_radians = np.radians(angles)

# Calculer cosinus et sinus pour éviter la division par zéro
cos_angles = np.cos(angles_radians)
sin_angles = np.sin(angles_radians)

# Définir le rectangle
demi_largeur, demi_hauteur = 0.8, 0.5
fig, ax = plt.subplots()
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect(1.0)
rectangle = plt.Rectangle((-demi_largeur, -demi_hauteur), 2 * demi_largeur, 2 * demi_hauteur, fill=False)
ax.add_artist(rectangle)

# Générer des cercles
rayons = np.random.uniform(0.1, 0.3, 3)
positions_x = np.random.uniform(-1 + 0.3, 1 - 0.3, 3)
positions_y = np.random.uniform(-1 + 0.3, 1 - 0.3, 3)
for px, py, r in zip(positions_x, positions_y, rayons):
    circle = plt.Circle((px, py), r, color='r', fill=False, linestyle='--')
    ax.add_artist(circle)

# Calculer les intersections pour chaque rayon
for angle, cos_a, sin_a in zip(angles, cos_angles, sin_angles):
    ray_min_t = np.inf  # Commencer avec une très grande distance
    for px, py, r in zip(positions_x, positions_y, rayons):
        calc_disc = 4*(r**2 - (sin_a*(-px) - cos_a*(-py))**2)
        if calc_disc >= 0:
            disc_value = np.sqrt(r**2 - (cos_a*py - sin_a*px)**2)
            t1 = cos_a*px + sin_a*py - disc_value
            t2 = cos_a*px + sin_a*py + disc_value
            if t1 > 0:
                ray_min_t = min(ray_min_t, t1)
            if t2 > 0:
                ray_min_t = min(ray_min_t, t2)
    
    # Si aucune intersection avec un cercle, prendre la distance au rectangle
    if ray_min_t == np.inf:
        ray_min_t = min(demi_largeur / np.abs(cos_a), demi_hauteur / np.abs(sin_a))
    
    x = ray_min_t * cos_a
    y = ray_min_t * sin_a
    ax.plot([0, x], [0, y], 'o-')

plt.show()
