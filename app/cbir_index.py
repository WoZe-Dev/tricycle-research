import os
import cv2
import numpy as np
import json
from imagededup.methods import PHash

# Fonction pour calculer l'histogramme de couleur
def calculate_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Fonction pour créer l'index des descripteurs d'images
def create_image_index(image_folder, output_file='image_index.json'):
    index = {}

    for image_name in os.listdir(image_folder):
        if image_name.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, image_name)
            image = cv2.imread(image_path)
            if image is None:
                continue

            # Calculer l'histogramme de couleur
            color_histogram = calculate_color_histogram(image)

            # Ajouter l'entrée à l'index
            index[image_name] = color_histogram.tolist()

    # Sauvegarder l'index dans un fichier JSON
    with open(output_file, 'w') as f:
        json.dump(index, f)

if __name__ == "__main__":
    image_folder = 'images_db'  # Dossier contenant toutes les images
    create_image_index(image_folder)
