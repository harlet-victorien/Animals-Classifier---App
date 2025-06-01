import streamlit as st
from globals import COLORS

def create_sidebar():
    """Create sidebar with configuration options"""
    st.sidebar.header("Configuration")
    
    # File upload section
    file_path = create_file_upload_section()
    
    # Analysis parameters
    analysis_params = create_analysis_parameters_section()
    
    # Optimization ranges
    optimization_ranges = create_optimization_ranges_section()
    
    # Combine all config
    config = {
        'file_path': file_path,
        **analysis_params,
        **optimization_ranges
    }
    
    return config

def create_file_upload_section():
    """Create file upload section"""
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload a CSV file with investment data"
    )
    
    if uploaded_file is None:
        file_path = "data/test.csv"
        st.sidebar.info("Using default test.csv file")
    else:
        # Save uploaded file temporarily
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    return file_path

def create_analysis_parameters_section():
    """Create analysis parameters section"""
    st.sidebar.subheader("Analysis Parameters")
    
    stop_loss = st.sidebar.slider(
        "Stop Loss", 
        0.0, 1.0, 0.3, 0.1, 
        help="Stop loss percentage"
    )
    
    with_scam = st.sidebar.checkbox(
        "Include Scam Data", 
        False, 
        help="Include scam data in analysis"
    )
    
    return {
        'stop_loss': stop_loss,
        'with_scam': with_scam
    }

def create_optimization_ranges_section():
    """Create optimization ranges section"""
    st.sidebar.subheader("Optimization Ranges")
    
    x_min = st.sidebar.number_input("X Min", value=20000, step=10000)
    x_max = st.sidebar.number_input("X Max", value=1000000, step=10000)
    y_min = st.sidebar.number_input("Y Min", value=20000, step=10000)
    y_max = st.sidebar.number_input("Y Max", value=1000000, step=10000)
    step = st.sidebar.number_input("Step Size", value=10000, step=1000)
    
    return {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'step': step
    }