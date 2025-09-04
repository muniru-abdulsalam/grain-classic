import os
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.activations import swish
import plotly.express as px
import pandas as pd
import gdown # New library to download from Google Drive

# =========================
# File Paths for Models
# =========================
# Google Drive IDs for the models
MILLET_ID = "1NMYkFxQRSOoZLa3BkANfAN3Rk7vQNJw-"
MAIZE_ID = "1ZSuW7UGHe_33M9O-LPygE1sTIF1iDZYV"

# Local paths to save the downloaded models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MILLET_PATH = os.path.join(MODEL_DIR, "model_checkpoint_6.h5")
MAIZE_PATH = os.path.join(MODEL_DIR, "maize_model_local.h5")

# Function to download models from Google Drive
def download_model_from_gdrive(file_id, output_path):
    if not os.path.exists(output_path):
        try:
            gdown.download(id=file_id, output=output_path, quiet=False)
            return True
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            return False
    return True

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Grain Variety Classifier",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Custom CSS for Better Styling
# =========================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .info-card {
        background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
        color: #2d5016;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .info-card h4 {
        color: #1b5e20;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    .upload-section {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f8fff8 0%, #e8f5e8 100%);
        margin: 1rem 0;
        color: #2e7d32;
    }
    
    .summary-card {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        color: #e65100;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .summary-card h3, .summary-card h4 {
        color: #bf360c;
        margin-bottom: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
        padding-left: 20px;
        padding-right: 20px;
        color: #424242;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white !important;
    }
    
    /* Ensure text visibility in dataframes */
    .stDataFrame {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Style for variety distribution text */
    .variety-stats {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        color: #0d47a1;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Custom activation support
# =========================
get_custom_objects().update({'swish': swish})

# =========================
# Load Models with Progress
# =========================
@st.cache_resource
def load_models():
    """Load ML models with caching for better performance"""
    models = {'millet': None, 'maize': None}
    
    with st.spinner('Loading AI models...'):
        # Check and download models
        millet_downloaded = download_model_from_gdrive(MILLET_ID, MILLET_PATH)
        maize_downloaded = download_model_from_gdrive(MAIZE_ID, MAIZE_PATH)
        
        if millet_downloaded:
            models['millet'] = load_model(MILLET_PATH, compile=False)
        
        if maize_downloaded:
            models['maize'] = load_model(MAIZE_PATH, compile=False)
            
    return models

models = load_models()

# =========================
# Class Information
# =========================
GRAIN_INFO = {
    'millet': {
        'classes': ['Finger Millet', 'Pearl Millet'],
        'target_size': (600, 600),
        'description': 'Millet is a highly nutritious grain that is drought-resistant and grows well in arid conditions.',
        'varieties': {
            'Finger Millet': 'Rich in calcium and amino acids, excellent for porridge and traditional foods.',
            'Pearl Millet': 'High in protein and iron, commonly used for making flour and traditional beverages.'
        }
    },
    'maize': {
        'classes': ['Bihilifa', 'Sanzal-sima', 'Wang Dataa'],
        'target_size': (224, 224),
        'description': 'Maize is a staple crop in Ghana, providing essential carbohydrates and nutrients.',
        'varieties': {
            'Bihilifa': 'A local variety known for its adaptability to local growing conditions.',
            'Sanzal-sima': 'Traditional variety with good storage properties and disease resistance.',
            'Wang Dataa': 'High-yielding variety popular among Ghanaian farmers.'
        }
    }
}

# =========================
# Prediction Function
# =========================
def predict_image(model, image, target_size, classes):
    """Enhanced prediction with confidence scoring"""
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = float(np.max(preds))
    
    # Get all predictions for visualization
    pred_data = []
    for i, class_name in enumerate(classes):
        pred_data.append({
            'Variety': class_name,
            'Confidence': float(preds[0][i]) * 100
        })
    
    return classes[class_idx], confidence, pred_data

def create_confidence_chart(pred_data):
    """Create a confidence visualization chart"""
    df = pd.DataFrame(pred_data)
    fig = px.bar(
        df, 
        x='Confidence', 
        y='Variety',
        orientation='h',
        color='Confidence',
        color_continuous_scale='Viridis',
        title='Prediction Confidence Scores'
    )
    fig.update_layout(
        height=300,
        showlegend=False,
        xaxis_title="Confidence (%)",
        yaxis_title="Grain Variety"
    )
    return fig

# =========================
# Sidebar Information
# =========================
with st.sidebar:
    st.markdown("### 📊 Model Status")
    
    if models['millet']:
        st.success("Millet Model: Ready")
    else:
        st.error("Millet Model: Not Found")
    
    if models['maize']:
        st.success("Maize Model: Ready")
    else:
        st.error("Maize Model: Not Found")
    
    st.markdown("---")
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. **Choose a grain type** (Millet or Maize)
    2. **Upload an image** of the grain
    3. **View the prediction** with confidence scores
    4. **Learn more** about the identified variety
    """)
    
    st.markdown("---")
    st.markdown("About the Models")
    st.markdown("""
    Our AI models are trained on thousands of grain images to accurately classify Ghanaian millet and maize varieties. The models use deep learning techniques to analyze visual features and provide reliable predictions.
    """)

# =========================
# Main Application
# =========================
st.markdown("""
<div class="main-header">
    <h1>🌾 Ghanaian Grain Variety Classifier</h1>
    <p>AI-powered identification of millet and maize varieties</p>
</div>
""", unsafe_allow_html=True)

# Create tabs with enhanced styling
tab1, tab2, tab3 = st.tabs(["🌾 Millet Classification", "🌽 Maize Classification", "📚 Learn More"])

# =========================
# Millet Classification Tab
# =========================
with tab1:
    st.markdown("### Upload Millet Images")
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Choose millet images...", 
        type=["jpg", "jpeg", "png"], 
        key="millet_upload",
        accept_multiple_files=True,
        help="Upload one or more clear images of millet grains for classification"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} image(s) uploaded**")
        
        # Process each uploaded file
        for idx, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"---")
            st.markdown(f"#### Image {idx + 1}: {uploaded_file.name}")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Uploaded Image {idx + 1}", use_column_width=True)
            
            with col2:
                if models['millet']:
                    with st.spinner(f'Analyzing image {idx + 1}...'):
                        label, confidence, pred_data = predict_image(
                            models['millet'], 
                            image, 
                            GRAIN_INFO['millet']['target_size'], 
                            GRAIN_INFO['millet']['classes']
                        )
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>🎯 Prediction Result</h3>
                        <h2>{label}</h2>
                        <p>Confidence: {confidence*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show confidence chart
                    fig = create_confidence_chart(pred_data)
                    fig.update_layout(title=f'Confidence Scores - Image {idx + 1}')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show variety information
                    st.markdown(f"""
                    <div class="info-card">
                        <h4>About {label}</h4>
                        <p>{GRAIN_INFO['millet']['varieties'][label]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.error("Millet model is not available. Please check the model file path.")
        
        # Summary section for multiple files
        if len(uploaded_files) > 1 and models['millet']:
            st.markdown("---")
            st.markdown("### 📊 Batch Processing Summary")
            
            # Create summary data
            summary_data = []
            for idx, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file)
                label, confidence, _ = predict_image(
                    models['millet'], 
                    image, 
                    GRAIN_INFO['millet']['target_size'], 
                    GRAIN_INFO['millet']['classes']
                )
                summary_data.append({
                    'Image': f"Image {idx + 1}",
                    'Filename': uploaded_file.name,
                    'Predicted Variety': label,
                    'Confidence': f"{confidence*100:.1f}%"
                })
            
            # Display summary table
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Summary statistics
            variety_counts = summary_df['Predicted Variety'].value_counts()
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Variety Distribution:**")
                for variety, count in variety_counts.items():
                    st.write(f"• {variety}: {count} images")
            
            with col2:
                # Create pie chart for variety distribution
                fig_pie = px.pie(
                    values=variety_counts.values, 
                    names=variety_counts.index,
                    title="Variety Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)

# =========================
# Maize Classification Tab
# =========================
with tab2:
    st.markdown("### Upload Maize Images")
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Choose maize images...", 
        type=["jpg", "jpeg", "png"], 
        key="maize_upload",
        accept_multiple_files=True,
        help="Upload one or more clear images of maize grains for classification"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} image(s) uploaded**")
        
        # Process each uploaded file
        for idx, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"---")
            st.markdown(f"#### Image {idx + 1}: {uploaded_file.name}")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Uploaded Image {idx + 1}", use_column_width=True)
            
            with col2:
                if models['maize']:
                    with st.spinner(f'Analyzing image {idx + 1}...'):
                        label, confidence, pred_data = predict_image(
                            models['maize'], 
                            image, 
                            GRAIN_INFO['maize']['target_size'], 
                            GRAIN_INFO['maize']['classes']
                        )
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>🎯 Prediction Result</h3>
                        <h2>{label}</h2>
                        <p>Confidence: {confidence*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show confidence chart
                    fig = create_confidence_chart(pred_data)
                    fig.update_layout(title=f'Confidence Scores - Image {idx + 1}')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show variety information
                    st.markdown(f"""
                    <div class="info-card">
                        <h4>About {label}</h4>
                        <p>{GRAIN_INFO['maize']['varieties'][label]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.error("❌ Maize model is not available. Please check the model file path.")
        
        # Summary section for multiple files
        if len(uploaded_files) > 1 and models['maize']:
            st.markdown("---")
            st.markdown("### 📊 Batch Processing Summary")
            
            # Create summary data
            summary_data = []
            for idx, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file)
                label, confidence, _ = predict_image(
                    models['maize'], 
                    image, 
                    GRAIN_INFO['maize']['target_size'], 
                    GRAIN_INFO['maize']['classes']
                )
                summary_data.append({
                    'Image': f"Image {idx + 1}",
                    'Filename': uploaded_file.name,
                    'Predicted Variety': label,
                    'Confidence': f"{confidence*100:.1f}%"
                })
            
            # Display summary table
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Summary statistics
            variety_counts = summary_df['Predicted Variety'].value_counts()
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Variety Distribution:**")
                for variety, count in variety_counts.items():
                    st.write(f"• {variety}: {count} images")
            
            with col2:
                # Create pie chart for variety distribution
                fig_pie = px.pie(
                    values=variety_counts.values, 
                    names=variety_counts.index,
                    title="Variety Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)

# =========================
# Learn More Tab
# =========================
with tab3:
    st.markdown("## 🌾 About Ghanaian Grains")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Millet Varieties")
        st.markdown(GRAIN_INFO['millet']['description'])
        
        for variety, description in GRAIN_INFO['millet']['varieties'].items():
            st.markdown(f"""
            <div class="info-card">
                <h4>{variety}</h4>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Maize Varieties")
        st.markdown(GRAIN_INFO['maize']['description'])
        
        for variety, description in GRAIN_INFO['maize']['varieties'].items():
            st.markdown(f"""
            <div class="info-card">
                <h4>{variety}</h4>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🔬 Technical Information")
    st.markdown("""
    - **Millet Model**: Trained on 600x600 pixel images
    - **Maize Model**: Trained on 224x224 pixel images
    - **Framework**: TensorFlow/Keras with deep learning
    - **Accuracy**: Models achieve high accuracy on validation datasets
    """)

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🌾 Ghanaian Grain Variety Classifier | Powered by AI & Machine Learning</p>
</div>
""", unsafe_allow_html=True)