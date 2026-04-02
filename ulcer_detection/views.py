import os
import numpy as np
import tensorflow as tf
import cv2
from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from .forms import FootUlcerForm  # Assuming you have a form for uploading images
from PIL import Image
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

# Load the trained model at the start
model_path = os.path.join(settings.BASE_DIR, 'ulcer_detection', 'ml_model', 'foot_ulcer_model.h5')
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading model from {model_path}: {e}")
    model = None

# Create a function to preprocess images for prediction
def preprocess_image(image_file):
    # Open the image using PIL
    img = Image.open(image_file)
    # Resize the image to match the input size expected by the model (224x224)
    img = img.resize((224, 224))
    # Convert image to numpy array
    img = np.array(img)
    # Convert from RGB to BGR (since OpenCV uses BGR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Normalize the image to [0, 1] range
    img = img / 255.0
    # Add an extra dimension to match the model input shape (batch size)
    img = np.expand_dims(img, axis=0)
    return img

# Prediction view to handle image upload and model prediction
def ulcer_detection_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_image = request.FILES['image']
        
        # Preprocess the image for prediction
        image = preprocess_image(uploaded_image)
        
        # Predict using the trained model
        if model:
            prediction = model.predict(image)
            # Determine result based on prediction
            result = 'Ulcer' if prediction[0][0] > 0.5 else 'No Ulcer'
        else:
            result = 'Model Error (The model file is missing or invalid)'
        
        # Save the uploaded image temporarily to display it
        image_path = default_storage.save(f'uploads/{uploaded_image.name}', ContentFile(uploaded_image.read()))
        image_url = default_storage.url(image_path)

        # Return the result and image URL to the template
        return render(request, 'result.html', {'result': result, 'image_url': image_url})

    # If GET request or invalid image, render the image upload form
    return render(request, 'upload.html')
