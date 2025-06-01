import streamlit as st
import torch
import os
import tempfile
import sys
from utils.predict_image import CNN

def load_default_model():
    """Load default model if it exists"""
    default_model_path = os.path.join(os.path.dirname(__file__), '..', 'utils/cifar10_cnn.pt')
    
    if os.path.exists(default_model_path):
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            state_dict = torch.load(default_model_path, map_location=device)
            
            model = CNN()
            model.load_state_dict(state_dict)
            model.eval()
            model.to(device)
            
            return model, device, 'cifar10_cnn.pt'
        except Exception as e:
            st.error(f"Error loading default model: {str(e)}")
            return None, None, None
    return None, None, None

def load_model_from_file(uploaded_file):
    """Load PyTorch model from uploaded file"""
    try:
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load the model state dict
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(tmp_file_path, map_location=device)
        
        # Initialize model using the existing CNN class
        model = CNN()
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def create_model_loader():
    """Create model loader tab content"""
    st.header("🤖 Model Loader")
    st.markdown(
        """
        Upload your PyTorch model file (.pth or .pt) to load it for inference.
        The model should be compatible with the CNN architecture used in this application.
        """
    )
    
    # Check if no model is loaded and try to load default
    if 'loaded_model' not in st.session_state:
        model, device, model_name = load_default_model()
        if model is not None:
            st.session_state['loaded_model'] = model
            st.session_state['model_device'] = device
            st.session_state['model_name'] = model_name
            st.info("🔄 Default model (cifar10_cnn.pt) loaded automatically!")
    
    # Default model section
    col_default, col_upload = st.columns(2)
    
    with col_default:
        st.subheader("🎯 Default Model")
        if st.button("🔄 Load Default Model (cifar10_cnn.pt)", type="secondary"):
            with st.spinner("Loading default model..."):
                model, device, model_name = load_default_model()
                if model is not None:
                    st.session_state['loaded_model'] = model
                    st.session_state['model_device'] = device
                    st.session_state['model_name'] = model_name
                    st.success("✅ Default model loaded successfully!")
                    st.info(f"Model is running on: {device}")
                    st.rerun()
                else:
                    st.error("❌ Default model (cifar10_cnn.pt) not found in utils folder.")
    
    with col_upload:
        st.subheader("📁 Upload Custom Model")
        # File uploader for model
        uploaded_model = st.file_uploader(
            "Choose a PyTorch model file",
            type=['pth', 'pt'],
            help="Upload a .pth or .pt file containing your trained model"
        )
    
    if uploaded_model is not None:
        st.success(f"Model file '{uploaded_model.name}' uploaded successfully!")
        
        # Show file details
        file_details = {
            "Filename": uploaded_model.name,
            "File size": f"{uploaded_model.size / (1024*1024):.2f} MB",
            "File type": uploaded_model.type
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📄 File Details")
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")
        
        with col2:
            st.subheader("🔄 Load Model")
            if st.button("Load Model", type="primary"):
                with st.spinner("Loading model..."):
                    model, device = load_model_from_file(uploaded_model)
                    
                    if model is not None:
                        # Store model in session state
                        st.session_state['loaded_model'] = model
                        st.session_state['model_device'] = device
                        st.session_state['model_name'] = uploaded_model.name
                        
                        st.success("✅ Model loaded successfully!")
                        st.info(f"Model is running on: {device}")
                        st.rerun()
                    else:
                        st.error("❌ Failed to load model. Please check the file format.")
    
    # Display current loaded model status with full information
    if 'loaded_model' in st.session_state:
        st.divider()
        st.subheader("📊 Currently Loaded Model")
        
        # Model metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Name", st.session_state.get('model_name', 'Unknown'))
        with col2:
            st.metric("Device", str(st.session_state.get('model_device', 'Unknown')))
        with col3:
            # Count total parameters
            model = st.session_state['loaded_model']
            total_params = sum(p.numel() for p in model.parameters())
            st.metric("Total Parameters", f"{total_params:,}")
        
        # Model architecture display
        st.subheader("🏗️ Model Architecture")
        with st.expander("View Model Architecture Details", expanded=False):
            st.code(str(model), language="python")
            
            # Parameter breakdown
            st.subheader("📈 Parameter Breakdown")
            param_info = []
            for name, param in model.named_parameters():
                param_info.append({
                    "Layer": name,
                    "Shape": str(list(param.shape)),
                    "Parameters": param.numel()
                })
            
            if param_info:
                st.table(param_info)
            
            # Model summary
            st.subheader("📋 Model Summary")
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            non_trainable_params = total_params - trainable_params
            
            summary_info = {
                "Total Parameters": f"{total_params:,}",
                "Trainable Parameters": f"{trainable_params:,}",
                "Non-trainable Parameters": f"{non_trainable_params:,}",
                "Model Size (approx)": f"{total_params * 4 / (1024*1024):.2f} MB"
            }
            
            for key, value in summary_info.items():
                st.write(f"**{key}:** {value}")
        
        # Control buttons
        col_clear, col_reload = st.columns(2)
        with col_clear:
            if st.button("🗑️ Clear Model", help="Remove the currently loaded model"):
                del st.session_state['loaded_model']
                del st.session_state['model_device']
                del st.session_state['model_name']
                st.success("✅ Model cleared successfully!")
                st.rerun()
        
        with col_reload:
            if st.button("🔄 Reload Model", help="Reload the current model"):
                if st.session_state.get('model_name') == 'cifar10_cnn.pt':
                    model, device, model_name = load_default_model()
                    if model is not None:
                        st.session_state['loaded_model'] = model
                        st.session_state['model_device'] = device
                        st.session_state['model_name'] = model_name
                        st.success("✅ Default model reloaded successfully!")
                        st.rerun()
                else:
                    st.warning("⚠️ Cannot reload uploaded models. Please re-upload the file.")
    else:
        st.info("ℹ️ No model loaded. Please upload a model file above or use the default model.")