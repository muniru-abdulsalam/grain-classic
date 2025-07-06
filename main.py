import os
import gdown
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from keras.models import load_model
from streamlit_option_menu import option_menu
from keras.preprocessing.image import img_to_array

# === Streamlit Config ===
st.set_page_config(page_title="Dashboard", layout="wide")
st.sidebar.image("assets/logo.png", caption='WELCOME')

# === Load Both Models ===
# MILLET_MODEL_PATH = r"C:\Users\HP\OneDrive\Desktop\FINAL\model_checkpoint_6.h5"
# MAIZE_MODEL_PATH = r"C:\Users\HP\OneDrive\Desktop\FINAL\maize_millet_model_vgg16_2.h5"

# millet_model = load_model(MILLET_MODEL_PATH, compile=False)
# maize_model = load_model(MAIZE_MODEL_PATH, compile=False)

MODEL_DIR = "models"
MILLET_ID = "1NMYkFxQRSOoZLa3BkANfAN3Rk7vQNJw-"
MAIZE_ID = "1khpZJO5MI-RB8sKhUHAA_-fMaSNfSkaH"

MILLET_PATH = os.path.join(MODEL_DIR, "millet_model.h5")
MAIZE_PATH = os.path.join(MODEL_DIR, "maize_model.h5")

# === Ensure model directory exists ===
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# === Download models if not already present ===
if not os.path.exists(MILLET_PATH):
    gdown.download(f"https://drive.google.com/uc?id={MILLET_ID}", MILLET_PATH, quiet=False)

if not os.path.exists(MAIZE_PATH):
    gdown.download(f"https://drive.google.com/uc?id={MAIZE_ID}", MAIZE_PATH, quiet=False)

# === Load models ===
millet_model = load_model(MILLET_PATH, compile=False)
maize_model = load_model(MAIZE_PATH, compile=False)

# === Class Mappings ===
millet_mappings = {0: 'Finger millet', 1: 'Pearl millet'}
maize_mappings = {0: 'Bihilifa', 1: 'SanzalSima', 2: 'WangDataa'}

# === Select Model Type ===
# === Select Model Type ===
model_type = st.sidebar.radio("Select Model", options=["Millet", "Maize"])
st.sidebar.markdown("ℹ️ **Millet**: Finger & Pearl millet  \n**Maize**: Bihilifa, SanzalSima, WangDataa")

if model_type == "Millet":
    model = millet_model
    mappings = millet_mappings
    target_size = (600, 600)
else:
    model = maize_model
    mappings = maize_mappings
    target_size = (224, 224)


# === Upload Single Image ===
def Upload():
    st.title("Upload a Single Image")
    st.markdown("---")
    st.header("Image File Upload")
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

    col1, col2 = st.columns(2)
    predict = False
    model_image = np.array([])

    with col1:
        if uploaded_file is not None:
            st.success("Image uploaded successfully!")
            image = Image.open(uploaded_file)
            model_image = image.copy()
            display_image = cv2.resize(np.array(image), (400, 400))
            st.image(display_image, caption='Uploaded Image')
            predict = st.button(label="Click Me for Prediction")

    with col2:
        if predict:
            model_image = model_image.resize(target_size)
            img_array = img_to_array(model_image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = float(predictions[0][predicted_class]) * 100
            predicted_label = mappings[predicted_class]
            st.success(f"Predicted Class: **{predicted_label}**  \n Confidence: **{confidence:.2f}%**")



# === Upload Multiple Images ===

import pandas as pd

def Uploads():
    st.title("Upload Multiple Images")
    st.markdown("---")
    uploaded_files = st.file_uploader(
        "Choose image files", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} image(s) uploaded successfully.")
        st.write("Preview")

        predict = st.button("Click Me for Prediction")

        if predict:
            num_columns = 3
            images = [Image.open(file) for file in uploaded_files]
            num_rows = len(images) // num_columns + int(len(images) % num_columns != 0)

            results = []
            progress_bar = st.progress(0)

            for row in range(num_rows):
                cols = st.columns(num_columns)
                for col, (file, image) in zip(
                    cols,
                    zip(
                        uploaded_files[row * num_columns:(row + 1) * num_columns],
                        images[row * num_columns:(row + 1) * num_columns]
                    )
                ):
                    with col:
                        # Resize and display image
                        display_image = cv2.resize(np.array(image), (400, 400))
                        st.image(display_image, caption=file.name)

                        # Predict
                        model_image = image.resize(target_size)
                        img_array = img_to_array(model_image) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)
                        predictions = model.predict(img_array)
                        predicted_class = np.argmax(predictions, axis=1)[0]
                        confidence = float(predictions[0][predicted_class]) * 100
                        predicted_label = mappings[predicted_class]

                        # Show prediction info under image
                        st.markdown(f"""
                            **Class:** {predicted_label}  
                            **Confidence:** {confidence:.2f}%
                        """)

                        results.append({
                            "Filename": file.name,
                            "Predicted Class": predicted_label,
                            "Confidence (%)": f"{confidence:.2f}"
                        })

                        progress = (len(results)) / len(uploaded_files)
                        progress_bar.progress(progress)

            # Display summary table
            if results:
                st.markdown("Summary Table")
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)

                # Download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name='prediction_results.csv',
                    mime='text/csv'
                )




# === Home Page ===
def Home():
    st.header('Hello, Welcome!')
    st.markdown(""" <div style= 'text-align: center; 
             font-size: 18px; color: white;'> 
             This is a <b>grain classifier</b> that helps you identify varieties of 
             <span style= 'color: yellow;'>Millet</span> and <span style= 'color: yellow;'>Maize</span> in Ghana!<br><br>
             <i>Enjoy your stay here!</i>
             </div> """, unsafe_allow_html=True)

# === Sidebar Navigation ===
def sideBar():
    with st.sidebar:
        selected = option_menu(
           menu_title='Main Menu',
           options=['Home', 'Upload', 'Uploads'],
           icons=["house", "file", "folder"],
           default_index=0,
           menu_icon='cast'
        )
    if selected == 'Home':
        Home()
    if selected == 'Upload':
        Upload()
    if selected == 'Uploads':
        Uploads()

# === Main ===
def main():
    sideBar()

if __name__ == '__main__':
    main()
