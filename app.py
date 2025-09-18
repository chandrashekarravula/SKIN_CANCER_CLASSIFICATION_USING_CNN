import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import os
import datetime

# Set page title and layout
st.set_page_config(page_title="Skin Cancer Classification", layout="wide")

# Load the saved model
@st.cache_resource  # Cache model for faster reloads
def load_model():
    return tf.keras.models.load_model('skin_cancer_model.h5')  # or .keras

model = load_model()

CLASS_NAMES = [
    "Actinic Keratosis",          # 0
    "Basal Cell Carcinoma",       # 1
    "Dermatofibroma",             # 2
    "Melanoma",                   # 3
    "Nevus",                      # 4
    "Pigmented Benign Keratosis", # 5
    "Seborrheic Keratosis",       # 6
    "Squamous Cell Carcinoma",    # 7
    "Vascular Lesion"             # 8
]


# Preprocess image for model input
def preprocess_image(image):
    img = image.resize((150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    predicted_class = CLASS_NAMES[class_idx]
    return predicted_class, prediction 

# Streamlit UI
st.title("Skin Cancer Classification using CNN 🏥")
st.write("Upload an image of a skin lesion for classification.")

# Add exit button in the sidebar
     # This will stop the script execution

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    # Preprocess and predict
    predicted_class, prediction = preprocess_image(image)
    # Show prediction
    st.subheader("Prediction Result")
    st.success(f"**Class:** {predicted_class}")
    
    if st.button("Know more") :
        if predicted_class == "Melanoma":
            st.title("MELANOMA :")
            st.header("Cause:")
            st.write("Caused by UV radiation (sun exposure/tanning beds) damaging melanocytes (pigment-producing cells) – often due to genetic mutations.")
            st.header("Effects:")
            st.write("Aggressive skin cancer that can metastasize (spread) if untreated – leading to life-threatening complications.")
            st.header("Precautions/Cure:")
            st.write("Early surgical removal is key. Use sunscreen, avoid excessive sun exposure, and monitor moles (ABCDE rule: Asymmetry, Border irregularity, Color variation, Diameter >6mm, Evolving).")
    
        elif predicted_class == "Nevus":
            st.title("MELANOCYTIC NEVUS (MOLE):")
            st.header("Cause:")
            st.write("Benign proliferation of melanocytes – often due to genetics or sun exposure.")
            st.header("Effects:")
            st.write("Usually harmless – but may rarely turn into melanoma (if dysplastic or changing).")
            st.header("Precautions/Cure:")
            st.write("Monitor for changes (size, color, shape). Suspicious moles should be biopsied/excised.")
        
        elif predicted_class == "Basal Cell Carcinoma":
            st.title("BASAL CELL CARCINOMA (BCC):")
            st.header("Cause:")
            st.write("UV-induced mutations in basal cells – most common skin cancer.")
            st.header("Effects:")
            st.write("Slow-growing, locally invasive – but rarely metastatic. Can cause disfigurement if untreated.")
            st.header("Precautions/Cure:")
            st.write("Surgical removal, cryotherapy, or topical treatments. Prevent with sun protection (hats, sunscreen).")
        
        elif predicted_class == "Actinic Keratosis":
            st.title("ACTINIC KERATOSIS (AK):")
            st.header("Cause:")
            st.write("Chronic sun exposure damaging keratinocytes – considered precancerous.")
            st.header("Effects:")
            st.write("Rough, scaly patches – may progress to squamous cell carcinoma (SCC).")
            st.header("Precautions/Cure:")
            st.write("Treated with cryotherapy, topical creams (5-FU, imiquimod), or photodynamic therapy. Sun avoidance is critical.")
        
        elif predicted_class == "Pigmented Benign Keratosis":
            st.title("PIGMENTED BENIGN KERATOSIS:")
            st.header("Cause:")
            st.write("Linked to aging and genetics – not related to UV exposure.")
            st.header("Effects:")
            st.write("Waxy, raised, non-cancerous lesions – no health risk but may be itchy or cosmetically bothersome.")
            st.header("Precautions/Cure:")
            st.write("Removal optional (cryotherapy, scraping). No prevention needed.")
        
        elif predicted_class == "Seborrheic Keratosis":
            st.title("SEBORRHEIC KERATOSIS:")
            st.header("Cause:")
            st.write("Non-cancerous skin growth – associated with aging and genetics.")
            st.header("Effects:")
            st.write("Brown, black, or light tan growths – usually harmless and not linked to sun exposure.")
            st.header("Precautions/Cure:")
            st.write("No treatment needed unless symptomatic. Can be removed for cosmetic purposes.")
        
        elif predicted_class == "Dermatofibroma":
            st.title("DERMATOFIBROMA:")
            st.header("Cause:")
            st.write("Benign fibrous tumor – often from minor trauma (e.g., insect bite).")
            st.header("Effects:")
            st.write("Firm, pigmented nodule – harmless but may be tender.")
            st.header("Precautions/Cure:")
            st.write("Usually left untreated – excision if symptomatic.")
        
        elif predicted_class == "Squamous Cell Carcinoma":
            st.title("SQUAMOUS CELL CARCINOMA (SCC):")
            st.header("Cause:")
            st.write("Chronic UV exposure – mutations in squamous cells of the skin.")
            st.header("Effects:")
            st.write("Second most common skin cancer – can grow and spread if untreated.")
            st.header("Precautions/Cure:")
            st.write("Early surgical removal or radiation. Prevent with consistent sun protection.")
        
        elif predicted_class == "Vascular Lesion":
            st.title("VASCULAR LESION (e.g., Hemangioma, Angioma):")
            st.header("Cause:")
            st.write("Abnormal blood vessel growth – congenital or acquired.")
            st.header("Effects:")
            st.write("Red/purple marks – may bleed if traumatized (rarely serious).")
            st.header("Precautions/Cure:")
            st.write("Laser therapy for cosmetic concerns. Monitor for changes.")
        
        else:
            st.warning("Unknown condition. Please consult a dermatologist for further evaluation.")

# Footer
st.markdown("---")
current_year = datetime.datetime.now().year
footer_text = f"© {current_year} Chandrashekar Ravula. All rights reserved."
st.markdown(f"<div style='text-align: center; color: gray; font-size: 0.9rem;'>{footer_text}</div>", unsafe_allow_html=True)

