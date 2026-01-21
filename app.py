import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- 1. GLOBAL VARIABLES (Must be at the top) ---
class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
               'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
               'River', 'SeaLake']

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Satellite Vision",
    page_icon="🛰️",
    layout="centered"
)

# --- 3. LOAD MODEL FUNCTION ---
@st.cache_resource
def load_model():
    # Load the V2 model you uploaded to GitHub
    return tf.keras.models.load_model('my_satellite_model_v2.keras')

model = load_model()

# --- 4. SIDEBAR (Control Panel) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1995/1995671.png", width=100)
    st.title("Mission Control")
    st.info("This tool uses a Convolutional Neural Network (CNN) to classify terrain from satellite imagery.")
    
    # File Uploader
    file = st.file_uploader("Upload Satellite Image", type=["jpg", "png", "jpeg"])

# --- 5. MAIN PAGE (Results) ---
st.title("🛰️ Satellite Terrain Analysis")

if file is not None:
    # Create two columns for a professional look
    col1, col2 = st.columns(2)
    
    # Open and display the image
    img = Image.open(file)
    
    with col1:
        st.image(img, caption='Source Image', use_column_width=True)
        
    # Preprocess the image for the AI (Resize to 64x64)
    img_resized = img.resize((64, 64))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make Prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    # Calculate results
    confidence = 100 * np.max(score)
    predicted_class = class_names[np.argmax(score)]
    
    # Display Results in the second column
    with col2:
        st.subheader("Analysis Result")
        
        # Color-coded result based on confidence
        if confidence > 85:
            st.success(f"**{predicted_class}**")
        else:
            st.warning(f"**{predicted_class}**")
            
        st.metric(label="Confidence Level", value=f"{confidence:.2f}%")
        
        # Progress bar
        st.progress(int(confidence))

else:
    # Placeholder text when no image is uploaded
    st.write("👈 Upload an image in the sidebar to begin analysis.")
