import os
import cv2
import json
import requests
import numpy as np
from PIL import Image
import imagehash
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote

# Fonction pour télécharger une image
def download_image(url):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            return None
    except Exception as e:
        print(f"Erreur lors du téléchargement de l'image : {e}")
        return None

# Fonction pour recadrer automatiquement une image
def auto_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 10, 100)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return image[y:y+h, x:x+w]
    return image

# Fonction pour obtenir la couleur dominante d'une image
def get_dominant_color(image, k=1):
    pixels = np.float32(image.reshape(-1, 3))
    n_colors = k
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    dominant_color = palette[np.argmax(np.bincount(labels.flatten()))]
    return dominant_color.tolist()

# Fonction pour détecter les points d'intérêt avec ORB et extraire les descripteurs
def get_keypoints_and_descriptors(image):
    orb = cv2.ORB_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors

# Fonction pour comparer les descripteurs de deux images
def compare_descriptors(desc1, desc2):
    if desc1 is None or desc2 is None:
        return float('inf')  # Si l'une des images n'a pas de descripteur, renvoyer une grande distance
    
    # Utiliser BFMatcher pour comparer les descripteurs
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    
    # Calculer la distance moyenne des correspondances
    if len(matches) == 0:
        return float('inf')
    distances = [match.distance for match in matches]
    return np.mean(distances)

# Fonction pour parcourir les URLs des images et obtenir toutes les images
def get_image_urls_from_directory(base_url, max_depth=3):
    image_urls = []
    try:
        stack = [(base_url, 0)]
        while stack:
            current_url, depth = stack.pop()
            if depth > max_depth:
                continue
            response = requests.get(current_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.find_all('a'):
                    href = link.get('href')
                    full_url = urljoin(current_url, quote(href))
                    if href.endswith('/') and depth + 1 < max_depth:
                        stack.append((full_url, depth + 1))
                    elif href.endswith(('.jpg', '.png', '.jpeg')):
                        image_urls.append(full_url)
    except Exception as e:
        print(f"Erreur lors du parcours du dossier : {e}")
    
    return image_urls

# Fonction principale pour calculer les hachages, les couleurs dominantes et les descripteurs
def calculate_and_store_image_index(base_url, output_file='image_index.json'):
    image_urls = get_image_urls_from_directory(base_url)
    image_index = []

    for img_url in image_urls:
        img_database = download_image(img_url)
        if img_database is not None:
            img_database = auto_crop(img_database)
            _, descriptors = get_keypoints_and_descriptors(img_database)
            dominant_color = get_dominant_color(img_database)
            image_index.append({
                'url': img_url,
                'descriptors': descriptors.tolist() if descriptors is not None else None,
                'dominant_color': dominant_color
            })

    # Enregistrer les résultats dans un fichier JSON
    with open(output_file, 'w') as f:
        json.dump(image_index, f)
    
    print(f"Index des images généré et enregistré dans {output_file}")

# Comparaison avec l'index
def compare_uploaded_image_with_index(uploaded_image_path, index_file='image_index.json'):
    img_uploaded = cv2.imread(uploaded_image_path)
    if img_uploaded is None:
        print("Erreur : L'image téléchargée n'a pas pu être lue.")
        return None, None
    img_uploaded = auto_crop(img_uploaded)
    _, uploaded_descriptors = get_keypoints_and_descriptors(img_uploaded)
    uploaded_dominant_color = get_dominant_color(img_uploaded)

    with open(index_file, 'r') as f:
        image_index = json.load(f)

    best_match = None
    best_score = float('inf')

    for image_data in image_index:
        if image_data['descriptors'] is None:
            continue

        descriptors = np.array(image_data['descriptors'], dtype=np.uint8)
        color_distance = np.linalg.norm(np.array(uploaded_dominant_color) - np.array(image_data['dominant_color']))
        descriptor_distance = compare_descriptors(uploaded_descriptors, descriptors)

        # Pondération : 70% pour les descripteurs, 30% pour la couleur
        combined_score = 0.7 * descriptor_distance + 0.3 * color_distance

        if combined_score < best_score:
            best_score = combined_score
            best_match = image_data['url']

    return best_match, best_score

if __name__ == "__main__":
    base_url = "http://51.83.79.4/img/"  # Assurez-vous que c'est la bonne URL
    calculate_and_store_image_index(base_url)

    