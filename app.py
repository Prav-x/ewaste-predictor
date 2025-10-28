import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
from ewaste_predictor import EWastePredictor
import plotly.express as px
import plotly.graph_objects as go

# Page configuration with modern settings
st.set_page_config(
    page_title="E-Waste AI Predictor & Recycling Guide",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.streamlit.io/',
        'Report a bug': "https://github.com/streamlit/streamlit/issues",
        'About': "# E-Waste AI Predictor\nTransforming e-waste management with AI."
    }
)

# Custom CSS for a modern, professional UI with animations
st.markdown("""
<style>
    /* Global styles with animations */
    body {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        animation: fadeIn 1s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Header styles with animation */
    .main-header {
        font-size: 3.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: slideDown 1.2s ease-out;
    }
    
    @keyframes slideDown {
        from { transform: translateY(-50px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #228B22;
        margin: 1rem 0;
        font-weight: 500;
        animation: fadeInUp 1.5s ease-out;
    }
    
    @keyframes fadeInUp {
        from { transform: translateY(30px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    /* Card styles with hover animations */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: bounceIn 1.8s ease-out;
    }
    .prediction-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }
    
    @keyframes bounceIn {
        0% { transform: scale(0.3); opacity: 0; }
        50% { transform: scale(1.05); }
        70% { transform: scale(0.9); }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .recycling-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 20px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: slideInLeft 2s ease-out;
    }
    .recycling-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-100px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 6px solid #2E8B57;
        transition: all 0.3s ease;
        animation: fadeInUp 2.2s ease-out;
    }
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    
    /* Button styles with animations */
    .stButton>button {
        background: linear-gradient(135deg, #2E8B57 0%, #228B22 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(46, 139, 87, 0.3);
        animation: pulse 2.5s infinite;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(46, 139, 87, 0.4);
        background: linear-gradient(135deg, #228B22 0%, #2E8B57 100%);
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 4px 15px rgba(46, 139, 87, 0.3); }
        50% { box-shadow: 0 4px 20px rgba(46, 139, 87, 0.5); }
        100% { box-shadow: 0 4px 15px rgba(46, 139, 87, 0.3); }
    }
    
    /* Tab styles with animations */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: linear-gradient(135deg, #f1f3f4 0%, #e8eaed 100%);
        border-radius: 15px 15px 0 0;
        padding: 0.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        background-color: transparent;
        border-radius: 10px;
        color: #333;
        font-weight: 500;
        transition: all 0.3s ease;
        animation: fadeIn 2.5s ease-out;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2E8B57 0%, #228B22 100%);
        color: white;
        box-shadow: 0 5px 15px rgba(46, 139, 87, 0.3);
        transform: scale(1.05);
    }
    
    /* Sidebar styles */
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        animation: slideInRight 1.5s ease-out;
    }
    
    @keyframes slideInRight {
        from { transform: translateX(100px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Metric cards with animations */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        text-align: center;
        transition: all 0.3s ease;
        animation: zoomIn 2.8s ease-out;
    }
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    @keyframes zoomIn {
        from { transform: scale(0.8); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }
    
    /* Footer with animation */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1.5rem;
        background: linear-gradient(135deg, #2E8B57 0%, #228B22 100%);
        color: white;
        border-radius: 15px;
        animation: fadeInUp 3s ease-out;
    }
    
    /* Loading spinner animation */
    .stSpinner > div > div {
        border-color: #2E8B57 transparent transparent transparent;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Progress bar for training */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #2E8B57 0%, #228B22 100%);
        animation: progress 2s ease-in-out;
    }
    
    @keyframes progress {
        0% { width: 0%; }
        100% { width: 100%; }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model with caching for performance"""
    predictor = EWastePredictor()
    
    # Attempt to load pre-trained model
    model_path = "ewaste_model.h5"
    if os.path.exists(model_path):
        predictor.load_model(model_path)
    else:
        # Fallback to creating a new model if none exists
        predictor.create_model()
        st.sidebar.warning("⚠️ No pre-trained model found. Using an untrained model. Train the model for better accuracy.")
    
    return predictor

def display_recycling_info(suggestions, predicted_class, confidence):
    """Display recycling information in a professional card format with animations"""
    st.markdown('<div class="recycling-card">', unsafe_allow_html=True)
    st.markdown(f"### ♻️ {predicted_class} Recycling Guide")
    st.markdown(f"**Confidence:** {confidence:.1%}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📋 Description")
        st.write(suggestions['description'])
        
        st.markdown("#### 🔄 Recycling Products")
        for i, product in enumerate(suggestions['recycling_products'], 1):
            st.write(f"{i}. {product}")
    
    with col2:
        st.markdown("#### ⚙️ Recycling Process")
        st.write(suggestions['recycling_process'])
        
        st.markdown("#### 🌱 Environmental Impact")
        st.write(suggestions['environmental_impact'])
    st.markdown('</div>', unsafe_allow_html=True)

def create_confidence_chart(predictions, class_names):
    """Create a modern bar chart for prediction confidence with animations"""
    # Sort predictions for better visualization
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_classes = [class_names[i] for i in sorted_indices]
    sorted_confidences = [predictions[i] for i in sorted_indices]
    
    fig = px.bar(
        x=sorted_confidences,
        y=sorted_classes,
        orientation='h',
        title="Prediction Confidence by E-Waste Type",
        labels={'x': 'Confidence', 'y': 'E-Waste Type'},
        color=sorted_confidences,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Confidence Score",
        yaxis_title="E-Waste Type",
        font=dict(size=14, family='Inter'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        transition=dict(duration=500, easing='cubic-in-out')
    )
    
    return fig

def main():
    # Sidebar for navigation and quick info with animation
    with st.sidebar:
        st.markdown("## 🗂️ Navigation")
        st.markdown("Select a tab to explore features.")
        st.markdown("---")
        st.markdown("### Quick Stats")
        if 'predictor' in locals() or 'predictor' in globals():
            predictor = load_model()
            st.write(f"**Classes:** {len(predictor.class_names)}")
        else:
            st.write("Model not loaded yet.")
    
    # Main header with animation
    st.markdown('<h1 class="main-header">♻️ E-Waste AI Predictor & Recycling Guide</h1>', unsafe_allow_html=True)
    st.markdown("### Empowering sustainable e-waste management with AI-driven classification and recycling insights.")
    
    # Load model
    predictor = load_model()
    
    # Modern tabs with icons and descriptions
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Predict E-Waste", 
        "📊 Model Training", 
        "📚 Recycling Guide", 
        "ℹ️ About"
    ])
    
    with tab1:
        st.markdown('<div class="sub-header">Upload an image to classify e-waste and receive tailored recycling guidance.</div>', unsafe_allow_html=True)
        
        # File uploader with enhanced UX
        uploaded_file = st.file_uploader(
            "Choose an image of e-waste",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image for accurate classification."
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True, output_format="auto")
            
            with col2:
                with st.spinner("Analyzing image..."):
                    try:
                        # Temporary save for prediction
                        temp_path = "temp_image.jpg"
                        image.save(temp_path)
                        
                        # Perform prediction
                        predicted_class, confidence, all_predictions = predictor.predict_ewaste(temp_path)
                        suggestions = predictor.get_recycling_suggestions(predicted_class)
                        
                        # Prediction result card with animation
                        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                        st.markdown("### 🎯 Prediction Result")
                        st.markdown(f"**Predicted Type:** {predicted_class}")
                        st.markdown(f"**Confidence:** {confidence:.1%}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Recycling info with animation
                        display_recycling_info(suggestions, predicted_class, confidence)
                        
                        # Confidence chart with animation
                        st.markdown("### 📊 Prediction Confidence Breakdown")
                        fig = create_confidence_chart(all_predictions, predictor.class_names)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Cleanup
                        os.remove(temp_path)
                        
                    except Exception as e:
                        st.error(f"❌ Error processing image: {str(e)}. Please try again with a different image.")
        else:
            st.info("👆 Upload an image to begin classification and get recycling suggestions.")
    
    with tab2:
        st.markdown('<div class="sub-header">Train and optimize the e-waste classification model.</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### ⚙️ Model Configuration")
            epochs = st.slider("Number of Epochs", 1, 100, 30, help="Higher epochs may improve accuracy but increase training time.")
            batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=1, help="Larger batches speed up training but require more memory.")
            patience = st.slider("Early Stopping Patience", 3, 20, 10, help="Stops training if no improvement to prevent overfitting.")
        
        with col2:
            st.markdown("#### 🚀 Training Status")
            if st.button("Start Training", type="primary", help="Initiate model training with current settings."):
                progress_bar = st.progress(0)
                with st.spinner("Training in progress... This may take several minutes."):
                    try:
                        predictor.batch_size = batch_size
                        predictor.create_model()
                        
                        history, test_gen = predictor.train_model(epochs=epochs, patience=patience)
                        report, cm, predictions = predictor.evaluate_model(test_gen)
                        predictor.save_model()
                        
                        st.success("✅ Training completed successfully!")
                        
                        # Results display with animations
                        st.markdown("#### 📈 Training Results")
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.metric("Final Training Accuracy", f"{history.history['accuracy'][-1]:.3f}")
                            st.metric("Final Validation Accuracy", f"{history.history['val_accuracy'][-1]:.3f}")
                        
                        with col_b:
                            st.metric("Final Training Loss", f"{history.history['loss'][-1]:.3f}")
                            st.metric("Final Validation Loss", f"{history.history['val_loss'][-1]:.3f}")
                        
                        # Training history chart with animation
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=history.history['accuracy'], name='Training Accuracy', line=dict(color='#2E8B57')))
                        fig.add_trace(go.Scatter(y=history.history['val_accuracy'], name='Validation Accuracy', line=dict(color='#FF6347')))
                        fig.update_layout(
                            title="Training History",
                            xaxis_title="Epoch",
                            yaxis_title="Accuracy",
                            font=dict(size=14, family='Inter'),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            transition=dict(duration=500, easing='cubic-in-out')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ An error occurred during training: {e}")
