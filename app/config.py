import streamlit as st
from globals import COLORS

def configure_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS globally
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Title and description
    st.title("Image Classifier Dashboard")
    
def get_custom_css():
    """Return custom CSS for styling all columns automatically"""
    # First, you need to convert the local image to base64
    import base64
    
    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    
    # Replace with your local image path
    try:
        base64_image = get_base64_image("assets/bg3.png")  # Put your image in assets folder
        background_image = f"data:image/png;base64,{base64_image}"
    except FileNotFoundError:
        # Fallback to the current online image
        background_image = "https://img.freepik.com/vecteurs-libre/vecteur-element-bulle-claire-fond-marine_53876-118551.jpg?semt=ais_hybrid&w=740"
    
    return f"""
        <style>
        /* Hide Streamlit header */
        header[data-testid="stHeader"] {{
            display: none !important;
        }}
        /* Apply image to header background */
        .stApp > div {{
            background-image: url('{background_image}');
            background-size: cover;
            background-position: center;
            background-color: {COLORS['backgroundColor']};
            color: {COLORS['textColor']};
            font-family: {COLORS['font']};
        }}
        
        /* Auto-apply borders to all Streamlit columns */
        .stColumn > div {{
            border: 1px solid {COLORS['borderColor']} !important;
            border-radius: 10px !important;
            background-color: {COLORS['secondaryBackgroundColor']} !important;
            padding: 20px !important;
            margin: 10px !important;
            min-height: 100px !important;
            color: {COLORS['textColor']} !important;
        }}
        
        /* Additional manual styling classes (keep for specific use cases) */
        .column-border {{
            border: 2px solid {COLORS['borderColor']};
            border-radius: 10px;
            padding: 20px;
            margin: 10px;
            background-color: {COLORS['secondaryBackgroundColor']};
            color: {COLORS['textColor']};
            box-shadow: 0 2px 4px rgba(78,138,230,0.3);
        }}
        .column-border-left {{
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 20px;
            margin: 10px;
            background-color: {COLORS['secondaryBackgroundColor']};
            color: {COLORS['textColor']};
            box-shadow: 0 2px 4px rgba(76,175,80,0.2);
        }}
        .column-border-right {{
            border: 2px solid {COLORS['borderColor']};
            border-radius: 10px;
            padding: 20px;
            margin: 10px;
            background-color: {COLORS['secondaryBackgroundColor']};
            color: {COLORS['textColor']};
            box-shadow: 0 2px 4px rgba(33,150,243,0.2);
        }}
        
        /* Make sure content inside columns doesn't inherit the border */
        .stColumn > div > div {{
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        
        /* Style text elements to use global colors */
        .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6 {{
            color: {COLORS['textColor']} !important;
        }}
        
        /* Style metric values */
        .metric-value {{
            color: {COLORS['primaryColor']} !important;
            font-weight: bold;
        }}
        
        /* Sidebar styling */
        .stSidebar {{
            background-color: {COLORS['secondaryBackgroundColor']} !important;
        }}
        
        .stSidebar > div:first-child {{
            background-color: {COLORS['secondaryBackgroundColor']} !important;
        }}
        
        /* Sidebar text color */
        .stSidebar .stMarkdown, .stSidebar .stText, .stSidebar h1, .stSidebar h2, .stSidebar h3 {{
            color: {COLORS['textColor']} !important;
        }}
        
        /* Sidebar input widgets */
        .stSidebar .stSelectbox label, .stSidebar .stSlider label, .stSidebar .stNumberInput label {{
            color: {COLORS['textColor']} !important;
        }}
                /* Button Colors */
        .stButton > button {{
            background-color: {COLORS['secondaryBackgroundColor']} !important;
            color: {COLORS['textColor']} !important;
            border: 1px solid {COLORS['borderColor']} !important;
            border-radius: 5px !important;
        }}
        
        .stButton > button:hover {{
            background-color: {COLORS['tertiaryBackgroundColor']} !important;
            color: {COLORS['backgroundColor']} !important;
            border-color: {COLORS['textColor']} !important;
            transition: background-color 0.5s ease, color 0.5s ease !important;
        }}
        
        .stButton > button:active {{
            background-color: {COLORS['backgroundColor']} !important;
            color: {COLORS['textColor']} !important;
        }}

        /* Tab styling */
        .stTabs [data-baseweb="tab-highlight"] {{
            background-color: {COLORS['primaryColor']} !important;
        }}

        </style>

        
        
    """