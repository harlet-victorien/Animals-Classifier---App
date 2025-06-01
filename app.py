import streamlit as st
from config import configure_page
from sidebar import create_sidebar
from tabs import create_tabs

def main():
    """Main application entry point"""
    # Configure page
    configure_page()

    
    # Create analyzer instance
    try:
        # Create tabs with analyzer and config
        create_tabs()
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()