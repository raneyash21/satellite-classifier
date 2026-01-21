import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 1. Load the Model ONCE (cache it so it's fast)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('my_satellite_model_v2.keras')

model = load_model()

# 2. Define Classes
class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
               'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
               'River', 'SeaLake']

# 3. Build the Website UI
st.title("🛰️ Satellite Image Classifier")
st.write("Upload a satellite image, and the AI will identify the terrain.")

# 4. The Upload Button
file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if file is not None:
    # Display the user's image
    img = Image.open(file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image for the AI
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make Prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    # Show Result
    confidence = 100 * np.max(score)
    predicted_class = class_names[np.argmax(score)]
    
    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Confidence: {confidence:.2f}%")