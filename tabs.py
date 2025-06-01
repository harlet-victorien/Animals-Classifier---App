import streamlit as st
from tabs_pages.model_loader import create_model_loader
from tabs_pages.model_inference import create_model_inference
from config import get_custom_css
from globals import COLORS

def create_tabs():
    """Create main tabs for the application"""
    
    # Apply custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs([
        "📊 Loading Model", 
        "🎯 Model Inference"
    ])
    
    with tab1:
        create_model_loader()
    
    with tab2:
        create_model_inference()
    

    
    # Footer
    create_footer()

def create_footer():
    """Create application footer"""
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: {COLORS['textColor']}'>
            <p>Dashboard built by HV</p>
        </div>
        """, 
        unsafe_allow_html=True
    )