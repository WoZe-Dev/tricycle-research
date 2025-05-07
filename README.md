# Comparaison d'images de produits

Il s'agit d'une application Web Flask simple permettant de comparer les images de produits téléchargées avec les images existantes stockées sur un serveur ou un api.

web : https://voxio.fr/tricycle-office-img

## Caractéristiques
- Téléchargez une image d'un produit.
- Comparez l'image téléchargée avec les images stockées à l'aide d'OpenCV (détection de fonctionnalités ORB)
- Renvoie si le produit est déjà dans la base de données ou non.

## comment exécuter :
1. Install : `pip install -r requirements.txt`
2. Run the Flask app: `python app.py`

## Requirements
- Python 3.x
- Flask
- OpenCV
- NumPy
- Requests
