import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- 1. PAGE CONFIGURATION (Must be the first command) ---
st.set_page_config(
    page_title="Satellite Vision",
    page_icon="🛰️",
    layout="centered" # or "wide"
)

# Load model function (same as before)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('my_satellite_model_v2.keras')

model = load_model()

class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
               'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
               'River', 'SeaLake']

# --- 2. SIDEBAR (The Control Panel) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1995/1995671.png", width=100)
    st.title("Mission Control")
    st.info("This tool uses a Convolutional Neural Network (CNN) to classify terrain from satellite imagery.")
    
    # Move the upload button here
    file = st.file_uploader("Upload Satellite Image", type=["jpg", "png", "jpeg"])

# --- 3. MAIN PAGE (The Results) ---
st.title("🛰️ Satellite Terrain Analysis")

if file is not None:
    # Use Columns to put Image and Result side-by-side
    col1, col2 = st.columns(2)
    
    img = Image.open(file)
    
    with col1:
        st.image(img, caption='Source Image', use_column_width=True)
        
    # Process image
    img_resized = img.resize((64, 64))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    confidence = 100 * np.max(score)
    predicted_class = class_names[np.argmax(score)]
    
    with col2:
        st.subheader("Analysis Result")
        # Visual Logic: Green for high confidence, Yellow for low
        if confidence > 85:
            st.success(f"**{predicted_class}**")
        else:
            st.warning(f"**{predicted_class}**")
            
        st.metric(label="Confidence Level", value=f"{confidence:.2f}%")
        
        # Add a progress bar for visual flair
        st.progress(int(confidence))

else:
    # Show a placeholder when no image is uploaded
    st.write("👈 Upload an image in the sidebar to begin analysis.")
    predicted_class = class_names[np.argmax(score)]
    
    st.success(f"Prediction: **{predicted_class}**")

    st.info(f"Confidence: {confidence:.2f}%")
