import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
from ewaste_predictor import EWastePredictor
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="E-Waste AI Predictor & Recycling Guide",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #228B22;
        margin: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .recycling-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E8B57;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model with caching"""
    predictor = EWastePredictor()
    
    # Try to load pre-trained model
    model_path = "ewaste_model.h5"
    if os.path.exists(model_path):
        predictor.load_model(model_path)
    else:
        # If no pre-trained model, create a new one
        predictor.create_model()
        st.warning("No pre-trained model found. Using untrained model. Please train the model first.")
    
    return predictor

def display_recycling_info(suggestions, predicted_class, confidence):
    """Display recycling information in an attractive format"""
    st.markdown(f'<div class="recycling-box">', unsafe_allow_html=True)
    st.markdown(f"### ♻️ {predicted_class} Recycling Guide")
    st.markdown(f"**Confidence:** {confidence:.1%}")
    
    st.markdown("#### 📋 Description")
    st.write(suggestions['description'])
    
    st.markdown("#### 🔄 Recycling Products")
    for i, product in enumerate(suggestions['recycling_products'], 1):
        st.write(f"{i}. {product}")
    
    st.markdown("#### ⚙️ Recycling Process")
    st.write(suggestions['recycling_process'])
    
    st.markdown("#### 🌱 Environmental Impact")
    st.write(suggestions['environmental_impact'])
    st.markdown('</div>', unsafe_allow_html=True)

def create_confidence_chart(predictions, class_names):
    """Create a bar chart showing prediction confidence for all classes"""
    # Sort by confidence
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
        yaxis_title="E-Waste Type"
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">♻️ E-Waste AI Predictor & Recycling Guide</h1>', unsafe_allow_html=True)
    st.markdown("### Transform your electronic waste into valuable resources with AI-powered classification and recycling suggestions")
    
    # Load model
    predictor = load_model()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Predict E-Waste", "📊 Model Training", "📚 Recycling Guide", "ℹ️ About"])
    
    with tab1:
        st.markdown('<div class="sub-header">Upload an image to predict e-waste type and get recycling suggestions</div>', unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image of e-waste",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of electronic waste to classify and get recycling suggestions"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                # Make prediction
                try:
                    # Save uploaded file temporarily
                    temp_path = "temp_image.jpg"
                    image.save(temp_path)
                    
                    # Predict
                    predicted_class, confidence, all_predictions = predictor.predict_ewaste(temp_path)
                    suggestions = predictor.get_recycling_suggestions(predicted_class)
                    
                    # Display prediction results
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown(f"### 🎯 Prediction Result")
                    st.markdown(f"**Predicted Type:** {predicted_class}")
                    st.markdown(f"**Confidence:** {confidence:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display recycling suggestions
                    display_recycling_info(suggestions, predicted_class, confidence)
                    
                    # Create confidence chart
                    st.markdown("### 📊 Prediction Confidence Breakdown")
                    fig = create_confidence_chart(all_predictions, predictor.class_names)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Clean up temp file
                    os.remove(temp_path)
                    
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
        else:
            st.info("👆 Please upload an image to get started!")
    
    with tab2:
        st.markdown('<div class="sub-header">Train the E-Waste Classification Model</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Model Configuration")
            epochs = st.slider("Number of Epochs", 1, 100, 30)
            batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=1)
            patience = st.slider("Early Stopping Patience", 3, 20, 10)
        
        with col2:
            st.markdown("#### Training Status")
            if st.button("🚀 Start Training", type="primary"):
                with st.spinner("Training model... This may take a while."):
                    try:
                        # Update predictor settings
                        predictor.batch_size = batch_size
                        predictor.create_model()
                        
                        # Train model
                        history, test_gen = predictor.train_model(epochs=epochs, patience=patience)
                        
                        # Evaluate model
                        report, cm, predictions = predictor.evaluate_model(test_gen)
                        
                        # Save model
                        predictor.save_model()
                        
                        st.success("✅ Model training completed successfully!")
                        
                        # Display training results
                        st.markdown("#### Training Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Final Training Accuracy", f"{history.history['accuracy'][-1]:.3f}")
                            st.metric("Final Validation Accuracy", f"{history.history['val_accuracy'][-1]:.3f}")
                        
                        with col2:
                            st.metric("Final Training Loss", f"{history.history['loss'][-1]:.3f}")
                            st.metric("Final Validation Loss", f"{history.history['val_loss'][-1]:.3f}")
                        
                        # Plot training history
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=history.history['accuracy'], name='Training Accuracy', line=dict(color='blue')))
                        fig.add_trace(go.Scatter(y=history.history['val_accuracy'], name='Validation Accuracy', line=dict(color='red')))
                        fig.update_layout(title="Training History", xaxis_title="Epoch", yaxis_title="Accuracy")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
        
        # Display dataset info
        st.markdown("#### 📁 Dataset Information")
        if os.path.exists("ewaste_dataset"):
            train_dir = "ewaste_dataset/train"
            if os.path.exists(train_dir):
                classes = os.listdir(train_dir)
                st.write(f"**Number of classes:** {len(classes)}")
                st.write(f"**Classes:** {', '.join(classes)}")
                
                # Count images per class
                class_counts = {}
                for cls in classes:
                    class_dir = os.path.join(train_dir, cls)
                    if os.path.isdir(class_dir):
                        class_counts[cls] = len(os.listdir(class_dir))
                
                # Create bar chart
                fig = px.bar(
                    x=list(class_counts.keys()),
                    y=list(class_counts.values()),
                    title="Training Images per Class",
                    labels={'x': 'E-Waste Type', 'y': 'Number of Images'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Training dataset not found!")
        else:
            st.error("Dataset directory not found!")
    
    with tab3:
        st.markdown('<div class="sub-header">Comprehensive E-Waste Recycling Guide</div>', unsafe_allow_html=True)
        
        # Display all recycling information
        for category, info in predictor.recycling_suggestions.items():
            with st.expander(f"♻️ {category}", expanded=False):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("**Description:**")
                    st.write(info['description'])
                    
                    st.markdown("**Recycling Products:**")
                    for i, product in enumerate(info['recycling_products'], 1):
                        st.write(f"{i}. {product}")
                
                with col2:
                    st.markdown("**Recycling Process:**")
                    st.write(info['recycling_process'])
                    
                    st.markdown("**Environmental Impact:**")
                    st.write(info['environmental_impact'])
    
    with tab4:
        st.markdown('<div class="sub-header">About the E-Waste AI Predictor</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 Purpose
        This AI-powered application helps classify electronic waste and provides detailed recycling suggestions to promote sustainable waste management and resource recovery.
        
        ### 🤖 Technology Stack
        - **Deep Learning:** TensorFlow/Keras with MobileNetV2 transfer learning
        - **Web Interface:** Streamlit for interactive user experience
        - **Data Visualization:** Plotly for dynamic charts and graphs
        - **Image Processing:** PIL and OpenCV for image preprocessing
        
        ### 🔬 Model Architecture
        - **Base Model:** MobileNetV2 (pre-trained on ImageNet)
        - **Classification Head:** Custom dense layers with dropout for regularization
        - **Data Augmentation:** Rotation, zoom, shift, and flip transformations
        - **Optimization:** Adam optimizer with learning rate scheduling
        
        ### 📊 Features
        1. **Image Classification:** Upload images to classify e-waste types
        2. **Recycling Guidance:** Get detailed recycling suggestions for each category
        3. **Model Training:** Train and evaluate the model with your dataset
        4. **Visualization:** Interactive charts showing prediction confidence
        5. **Environmental Impact:** Learn about the environmental benefits of recycling
        
        ### 🌱 Environmental Impact
        By properly classifying and recycling e-waste, we can:
        - Recover valuable materials (gold, silver, copper, rare earth elements)
        - Prevent toxic substances from contaminating soil and water
        - Reduce the need for mining new materials
        - Create jobs in the recycling industry
        - Reduce landfill waste
        
        ### 🚀 Getting Started
        1. Upload an image of electronic waste
        2. Get instant classification and recycling suggestions
        3. Learn about the recycling process and environmental benefits
        4. Train the model with your own dataset for better accuracy
        
        ### 📞 Support
        For questions or support, please refer to the documentation or contact the development team.
        """)
        
        # Display model info
        if predictor.model is not None:
            st.markdown("#### 🔧 Current Model Information")
            st.write(f"**Model Type:** Convolutional Neural Network (CNN)")
            st.write(f"**Base Architecture:** MobileNetV2")
            st.write(f"**Number of Classes:** {len(predictor.class_names)}")
            st.write(f"**Classes:** {', '.join(predictor.class_names)}")

if __name__ == "__main__":
    main()
