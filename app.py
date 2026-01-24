import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Satellite Terrain Analysis",
    page_icon="🛰️",
    layout="wide"
)

# --- HIDE STREAMLIT STYLE ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- LOAD THE MODEL (Cached for speed) ---
@st.cache_resource
def load_model():
    # Make sure 'satellite_model.h5' is the exact name of your saved file
    model = tf.keras.models.load_model('satellite_model.h5')
    return model

with st.spinner('Loading AI Model...'):
    model = load_model()

# --- CLASS NAMES ---
# UPDATE THIS LIST to match your exact training folders order!
class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 
    'Highway', 'Industrial', 'Pasture', 
    'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1995/1995655.png", width=80)
    st.title("Mission Control")
    st.info("This tool uses a Deep Learning model (CNN) to classify terrain from satellite imagery.")
    
    file = st.file_uploader("Upload Satellite Image", type=["jpg", "png", "jpeg"])

# --- MAIN PAGE ---
st.write("# 🛰️ Satellite Terrain Analysis")

def import_and_predict(image_data, model):
    # 1. Resize image to match model input (usually 64x64 or 256x256)
    size = (64, 64) # <--- CHANGE THIS to the size you used in training (e.g. 256, 256)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    
    # 2. Convert to array and normalize
    img = np.asarray(image)
    img = img / 255.0  # Normalize pixel values
    img_reshape = img[np.newaxis, ...] # Add batch dimension
    
    # 3. Predict
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.warning("⚠️ Upload an image in the sidebar to begin analysis.")
else:
    # Display the uploaded image
    image = Image.open(file)
    st.image(image, use_container_width=False, width=400, caption="Source Image")
    
    # Run Prediction
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    
    # --- 🛑 THE FIX: CONFIDENCE GATE ---
    # Convert highest score to percentage
    confidence = 100 * np.max(score)
    predicted_class = class_names[np.argmax(score)]

    # Threshold: If less than 85% confident, reject it.
    CONFIDENCE_THRESHOLD = 85.0

    if confidence < CONFIDENCE_THRESHOLD:
        st.error(f"❌ **Analysis Failed:** Low Confidence ({confidence:.2f}%)")
        st.warning(f"The model thinks this is **{predicted_class}**, but it's not sure.")
        st.info("💡 **Tip:** Please upload a clear satellite image (e.g., Forest, City, Highway). Standard photos (people, pets) will not work.")
    else:
        # --- SUCCESS: SHOW RESULT ---
        st.success("Analysis Complete")
        
        # Create 2 columns for the result metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Terrain Type", predicted_class)
        
        with col2:
            st.metric("Confidence Level", f"{confidence:.2f}%")
            st.progress(int(confidence))
