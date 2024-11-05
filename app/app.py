import os
import concurrent.futures
import requests
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote
import urllib3
import re
import logging
import fiftyone as fo

# Configurer Flask et FiftyOne
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DOWNLOADED_IMAGES'] = 'downloaded_images'
app.config['FEATURES_CACHE'] = 'features_cache'
app.secret_key = 'supersecretkey'

# Désactiver les avertissements SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configurer FiftyOne
if "image-comparison" in fo.list_datasets():
    fo.delete_dataset("image-comparison")
dataset = fo.Dataset(name="image-comparison")

# Configurer le logger
logging.basicConfig(level=logging.INFO)

# Charger le modèle VGG16 pour l'extraction de caractéristiques
model = VGG16(weights='imagenet', include_top=False, pooling='avg')

# Assurez-vous que les dossiers existent
os.makedirs(app.config['DOWNLOADED_IMAGES'], exist_ok=True)
os.makedirs(app.config['FEATURES_CACHE'], exist_ok=True)

# Supprimer tous les fichiers de cache existants pour forcer le recalcul
for f in os.listdir(app.config['FEATURES_CACHE']):
    os.remove(os.path.join(app.config['FEATURES_CACHE'], f))

# Fonction pour nettoyer le nom du fichier
def sanitize_filename(filename):
    return re.sub(r'[^\w\.-]', '', filename.replace(" ", "_"))

# Prétraitement avancé des images
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert("RGB")  # Convertir en RGB dès le départ
    img = img.resize((256, 256))  # Redimensionner à 256x256 pour cohérence
    img = ImageOps.exif_transpose(img)  # Supprimer les métadonnées
    img = ImageOps.grayscale(img)       # Conversion en niveaux de gris
    img = ImageOps.equalize(img)         # Égaliser l'histogramme pour uniformiser la luminosité et le contraste
    img = img.convert("RGB")             # Reconvertir en RGB
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = ImageEnhance.Brightness(img).enhance(1.2)
    return img

# Extraction et sauvegarde des caractéristiques d'une image
def extract_and_save_features(image_path):
    features_file = os.path.join(app.config['FEATURES_CACHE'], sanitize_filename(os.path.basename(image_path)) + ".npy")
    if not os.path.exists(features_file):
        img = preprocess_image(image_path)
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        features = model.predict(img_data).flatten()
        np.save(features_file, features)
    return features_file

# Charger les caractéristiques d'une image depuis le cache
def load_cached_features(image_path):
    features_file = os.path.join(app.config['FEATURES_CACHE'], sanitize_filename(os.path.basename(image_path)) + ".npy")
    if not os.path.exists(features_file):
        extract_and_save_features(image_path)
    return np.load(features_file)

# Calcul de la similarité entre deux vecteurs de caractéristiques
def calculate_similarity(features1, features2):
    return cosine_similarity([features1], [features2])[0][0]

# Téléchargement d'image si nécessaire
def download_image_if_needed(url, folder):
    filename = sanitize_filename(unquote(url).split("/")[-1])
    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        try:
            response = requests.get(url, stream=True, verify=False)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                extract_and_save_features(filepath)
                logging.info(f"Téléchargé et extrait : {filename}")
            else:
                logging.error(f"Erreur de téléchargement : {response.status_code} pour {url}")
        except Exception as e:
            logging.error(f"Erreur lors du téléchargement de {url} : {e}")
    return filepath

# Fonction pour récupérer récursivement les URLs des images
def fetch_image_urls(base_url, depth=0, max_depth=3):
    image_urls = []
    try:
        response = requests.get(base_url, verify=False)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            for link in soup.find_all('a'):
                href = link.get('href')
                if href:
                    full_url = urljoin(base_url, href)
                    if href.endswith('/') and depth < max_depth:
                        image_urls.extend(fetch_image_urls(full_url, depth + 1, max_depth))
                    elif href.lower().endswith(('jpg', 'jpeg', 'png', 'gif')):
                        image_urls.append(full_url)
        else:
            logging.error(f"Erreur d'accès à {base_url}: {response.status_code}")
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des URLs d'images depuis {base_url}: {e}")
    return image_urls

# Télécharger et traiter toutes les images au démarrage
def download_all_images():
    base_url = "51.83.79.4:8080/img/"
    image_urls = fetch_image_urls(base_url)
    logging.info(f"Nombre d'images trouvées : {len(image_urls)}")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(lambda url: download_image_if_needed(url, app.config['DOWNLOADED_IMAGES']), image_urls)
    logging.info("Téléchargement et traitement initial des images terminé.")

# Comparaison d'image avec ajustement dynamique du seuil
def compare_images(uploaded_image_path, downloaded_images_folder, min_results=10):
    uploaded_features = load_cached_features(uploaded_image_path)
    similarities = []
    print(f"Forme des caractéristiques extraites de l'image uploadée : {uploaded_features.shape}")
    for img_filename in os.listdir(downloaded_images_folder):
        if img_filename.endswith(('jpg', 'jpeg', 'png', 'gif')):
            img_path = os.path.join(downloaded_images_folder, img_filename)
            try:
                img_features = load_cached_features(img_path)
                if img_features.shape == uploaded_features.shape:
                    similarity = calculate_similarity(uploaded_features, img_features)
                    similarities.append((img_filename, similarity))
                    print(f"Similarité avec {img_filename} : {similarity}")
                else:
                    logging.warning(f"Dimension incompatible pour {img_filename}.")
            except Exception as e:
                logging.error(f"Erreur lors du traitement de l'image {img_filename}: {e}")

    if similarities:
        similarities.sort(key=lambda x: x[1], reverse=True)
        dynamic_threshold = max(0.4, np.median([s[1] for s in similarities]) - 0.1)
        filtered_similarities = [(filename, sim) for filename, sim in similarities if sim >= dynamic_threshold]
    else:
        logging.warning("Aucune similarité trouvée. Vérifiez les caractéristiques extraites.")
        filtered_similarities = []

    return filtered_similarities[:min_results]

# Routes Flask
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files or not request.files['file'].filename:
        flash('Aucun fichier sélectionné.')
        return redirect(url_for('index'))
    file = request.files['file']
    filename = sanitize_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    comparison_results = compare_images(filepath, app.config['DOWNLOADED_IMAGES'])
    return render_template('index.html', results=comparison_results)

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['DOWNLOADED_IMAGES'], filename)

# Lancement du serveur Flask
if __name__ == '__main__':
    download_all_images()
    app.run(host='192.168.1.145', port=5000, debug=True)
