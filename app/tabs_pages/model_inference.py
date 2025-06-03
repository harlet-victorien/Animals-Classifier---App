import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import tempfile
import os
import sys
from utils.predict_image import preprocess_image, class_names, class_names100
from globals import COLORS

def load_default_image():
    """Load default image if it exists"""
    default_image_path = os.path.join(os.path.dirname(__file__), '..', 'utils/cat.jpg')
    
    if os.path.exists(default_image_path):
        try:
            image = Image.open(default_image_path)
            return image, 'cat.jpg'
        except Exception as e:
            st.error(f"Error loading default image: {str(e)}")
            return None, None
    return None, None

def predict_image_with_confidence(model, image_path, device, class_names_list):
    """Modified version of predict_single_image to return confidence scores"""
    image = preprocess_image(image_path).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        confidence_scores, predicted_indices = torch.topk(probabilities, k=min(5, len(class_names_list)))
        
        # Convert to lists for easier handling
        confidence_scores = confidence_scores.squeeze().cpu().tolist()
        predicted_indices = predicted_indices.squeeze().cpu().tolist()
        
        # Handle single prediction case
        if not isinstance(confidence_scores, list):
            confidence_scores = [confidence_scores]
            predicted_indices = [predicted_indices]
        
        predictions = []
        for idx, conf in zip(predicted_indices, confidence_scores):
            predictions.append({
                'class': class_names_list[idx],
                'confidence': conf * 100
            })
    
    return predictions

def create_confidence_chart(predictions):
    """Create a bar chart for confidence scores"""
    classes = [pred['class'] for pred in predictions]
    confidences = [pred['confidence'] for pred in predictions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=confidences,
            y=classes,
            orientation='h',
            marker_color=COLORS['primaryColor'],
            text=[f"{conf:.1f}%" for conf in confidences],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Top Predictions with Confidence Scores",
        paper_bgcolor=COLORS['transparent'],
        plot_bgcolor=COLORS['transparent'],
        xaxis_title="Confidence (%)",
        yaxis_title="Classes",
        height=400,
        showlegend=False
    )
    
    return fig

def create_model_inference():
    """Create model inference tab content"""
    st.header("🎯 Model Inference")
    st.markdown(
        """
        Upload an image to get predictions from your loaded model. 
        The image will be automatically resized and preprocessed for inference.
        """
    )
    
    # Check if model is loaded
    if 'loaded_model' not in st.session_state:
        st.warning("⚠️ No model loaded. Please go to the Model Loader tab and load a model first.")
        return
    
    model = st.session_state['loaded_model']
    device = str(st.session_state['model_device'])
    
    # Create two columns for image upload and results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Image Upload")
        
        # Default image section
        col_default, col_upload = st.columns([1, 1])
        
        with col_default:
            if st.button("🎯 Use Default Image (cat.jpg)", type="secondary"):
                default_image, default_name = load_default_image()
                if default_image is not None:
                    st.session_state['current_image'] = default_image
                    st.session_state['current_image_name'] = default_name
                    st.success("✅ Default image loaded!")
                    st.rerun()
                else:
                    st.error("❌ Default image (cat.jpg) not found in project root.")
        
        with col_upload:
            # File uploader for images
            uploaded_image = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a PNG, JPG, or JPEG image file"
            )
        
        # Determine which image to display
        current_image = None
        image_name = "Unknown"
        
        if uploaded_image is not None:
            current_image = Image.open(uploaded_image)
            image_name = uploaded_image.name
            st.session_state['current_image'] = current_image
            st.session_state['current_image_name'] = image_name
        elif 'current_image' in st.session_state:
            current_image = st.session_state['current_image']
            image_name = st.session_state.get('current_image_name', 'Unknown')
        else:
            # Try to load default image automatically
            default_image, default_name = load_default_image()
            if default_image is not None:
                current_image = default_image
                image_name = default_name
                st.session_state['current_image'] = current_image
                st.session_state['current_image_name'] = image_name
                st.info("🔄 Default image (cat.jpg) loaded automatically!")
        
        if current_image is not None:
            # Display original image
            st.image(current_image, caption=f"Current Image: {image_name}", use_container_width=True)
            
            # Display image details
            st.write(f"**Image name:** {image_name}")
            st.write(f"**Image size:** {current_image.size}")
            st.write(f"**Image mode:** {current_image.mode}")
            
            # Show resized image
            # Crop to square by taking the center crop of the smaller dimension
            width, height = current_image.size
            min_dim = min(width, height)
            left = (width - min_dim) // 2
            top = (height - min_dim) // 2
            right = left + min_dim
            bottom = top + min_dim
            current_image = current_image.crop((left, top, right, bottom))
            resized_image = current_image.resize((32, 32))
            st.image(resized_image, caption="Resized Image (32x32)", width=150)
    
    with col2:
        st.subheader("🔍 Predictions")
        
        if 'current_image' in st.session_state:
            if st.button("🚀 Run Inference", type="primary"):
                with st.spinner("Making predictions..."):
                    try:
                        current_image = st.session_state['current_image']
                        
                        # Save current image to temporary file for processing
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                            current_image.save(tmp_file.name)
                            tmp_image_path = tmp_file.name
                        
                        # Make prediction using existing function
                        predictions = predict_image_with_confidence(model, tmp_image_path, device, class_names)
                        
                        # Clean up temporary file
                        os.unlink(tmp_image_path)
                        
                        # Display top prediction prominently
                        st.success(f"🎯 **Top Prediction:** {predictions[0]['class']} ({predictions[0]['confidence']:.1f}%)")
                        
                        # Display all predictions in a table
                        st.subheader("📊 Top 5 Predictions")
                        prediction_data = {
                            'Rank': range(1, len(predictions) + 1),
                            'Class': [pred['class'] for pred in predictions],
                            'Confidence (%)': [f"{pred['confidence']:.2f}" for pred in predictions]
                        }
                        
                        st.table(prediction_data)
                        
                        # Create and display confidence chart
                        fig = create_confidence_chart(predictions)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error during inference: {str(e)}")
        else:
            st.info("👆 Upload an image above or use the default image to see predictions here.")
    
    # Display model information
    st.divider()
    st.subheader("ℹ️ Model Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Name", st.session_state.get('model_name', 'Unknown'))
    with col2:
        st.metric("Device", device)
    with col3:
        st.metric("Classes", len(class_names))
    
    # Show example usage
    with st.expander("💡 Tips for better predictions"):
        st.markdown("""
        - **Image quality:** Use clear, well-lit images for better predictions
        - **Object centering:** Center the main object in the image
        - **Background:** Simple backgrounds often work better
        - **Size:** While images are automatically resized, higher quality originals may give better results
        - **Classes:** The model is trained on CIFAR-10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
        """)