import numpy as np
import matplotlib.pyplot as plt

def calculate_ray_circle_intersections(cos_angles, sin_angles, positions_x, positions_y, rayons, demi_largeur, demi_hauteur):
    distances = np.full(cos_angles.shape, np.inf)  # Initialement à l'infini
    
    for i, (cos_a, sin_a) in enumerate(zip(cos_angles, sin_angles)):
        for px, py, r in zip(positions_x, positions_y, rayons):
            # Calculer la distance du centre du cercle à l'origine
            center_to_origin = np.sqrt(px**2 + py**2)
            # Calculer la distance orthogonale du centre du cercle au rayon
            D_ortho = np.abs(px * sin_a - py * cos_a)
            # Vérifier si le cercle et le rayon s'intersectent
            if D_ortho <= r:
                # Calculer la distance le long du rayon jusqu'à la projection orthogonale du centre sur le rayon
                D_along_ray = (cos_a * px + sin_a * py)
                # Calculer la distance du centre du cercle à l'intersection avec le rayon
                D_to_intersection = np.sqrt(r**2 - D_ortho**2)
                # Calculer la distance totale du rayon à l'intersection
                distance = D_along_ray - D_to_intersection
                if distance > 0:
                    distances[i] = min(distances[i], distance)
        
        # Si aucune intersection n'a été trouvée, utiliser la distance par défaut au rectangle
        if distances[i] == np.inf:
            distances[i] = min(demi_largeur / np.abs(cos_a), demi_hauteur / np.abs(sin_a))
    
    return distances

def calculate_ray_circle_intersections_optimized(cos_angles, sin_angles, positions_x, positions_y, rayons, W_BORDER, H_BORDER):
    # Distance par défaut aux bordures
    default_distances = np.min(np.array([W_BORDER, H_BORDER]) / np.abs(np.vstack((cos_angles, sin_angles)).T), axis=1)
    
    # Calcul vectorisé de la distance orthogonale
    D_ortho = np.abs(np.outer(positions_x, sin_angles) - np.outer(positions_y, cos_angles))
    
    # Calcul vectorisé des intersections avec les cercles
    intersections = D_ortho <= rayons[:, None]
    
    # Calcul vectorisé de la distance le long du rayon jusqu'à la projection orthogonale
    D_along_ray = np.outer(positions_x, cos_angles) + np.outer(positions_y, sin_angles)
    
    # Calcul vectorisé de la distance du centre du cercle à l'intersection avec le rayon
    D_to_intersection = np.sqrt(np.maximum(rayons[:, None]**2 - D_ortho**2, 0))
    
    # Calcul vectorisé de la distance totale du rayon à l'intersection
    distances = np.where(intersections, D_along_ray - D_to_intersection, np.inf)
    
    # Trouver la distance minimale pour chaque rayon
    min_distances = np.min(np.where(distances > 0, distances, np.inf), axis=0)
    
    # Comparer avec la distance par défaut
    final_distances = np.minimum(min_distances, default_distances)
    
    return final_distances

# Paramètres pour le benchmark
N_RAYS = 360
W_BORDER = 0.8
H_BORDER = 0.5
rayons = np.random.uniform(0.1, 0.3, 30)
positions_x = np.random.uniform(-1 + 0.3, 1 - 0.3, 30)
positions_y = np.random.uniform(-1 + 0.3, 1 - 0.3, 30)
angles_radians = np.random.uniform(0, 2*np.pi, N_RAYS)
cos_angles = np.cos(angles_radians)
sin_angles = np.sin(angles_radians)

# Calcul et mesure du temps
import time
start_time = time.time()
distances_a = calculate_ray_circle_intersections_optimized(cos_angles, sin_angles, positions_x, positions_y, rayons, W_BORDER, H_BORDER)
end_time = time.time()
print(f"Temps d'exécution: {end_time - start_time:.4f} secondes")
start_time = time.time()
distances_b = calculate_ray_circle_intersections(cos_angles, sin_angles, positions_x, positions_y, rayons, W_BORDER, H_BORDER)
end_time = time.time()
print(f"Temps d'exécution: {end_time - start_time:.4f} secondes")
print(np.equal(distances_a, distances_b).all())







# # Paramètres
# angles = np.random.randint(0, 360, 15)
# angles_radians = np.radians(angles)
# cos_angles = np.cos(angles_radians)
# sin_angles = np.sin(angles_radians)
# demi_largeur, demi_hauteur = 0.8, 0.5

# # Définir le rectangle et les cercles
# rayons = np.random.uniform(0.1, 0.3, 3)
# positions_x = np.random.uniform(-1 + 0.3, 1 - 0.3, 3)
# positions_y = np.random.uniform(-1 + 0.3, 1 - 0.3, 3)

# # Calculer les distances d'intersection
# distances = calculate_ray_circle_intersections(cos_angles, sin_angles, positions_x, positions_y, rayons, demi_largeur, demi_hauteur)

# # Créer le plot
# fig, ax = plt.subplots()
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# ax.set_aspect(1.0)
# rectangle = plt.Rectangle((-demi_largeur, -demi_hauteur), 2 * demi_largeur, 2 * demi_hauteur, fill=False)
# ax.add_artist(rectangle)

# # Générer des cercles
# for px, py, r in zip(positions_x, positions_y, rayons):
#     circle = plt.Circle((px, py), r, color='r', fill=False, linestyle='--')
#     ax.add_artist(circle)

# # Dessiner les rayons et afficher les distances
# for angle, distance, cos_a, sin_a in zip(angles, distances, cos_angles, sin_angles):
#     x = distance * cos_a
#     y = distance * sin_a
#     ax.plot([0, x], [0, y], 'o-', color='blue')
#     ax.text(x, y, f"{distance:.2f}", fontsize=8, ha='right')

# plt.show()
