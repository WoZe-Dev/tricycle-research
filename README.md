# Product Image Comparison

This is a simple Flask web application to compare uploaded product images with existing images stored on a server.

## Features
- Upload an image of a product.
- Compare the uploaded image with stored images using OpenCV (ORB feature detection).
- Return whether the product is already in the database or not.

## How to run
1. Install the dependencies: `pip install -r requirements.txt`
2. Run the Flask app: `python app.py`

## Requirements
- Python 3.x
- Flask
- OpenCV
- NumPy
- Requests
