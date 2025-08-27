"""
Budget Table Extractor Application
Simple Streamlit app that redirects users to use Azure Blob Storage for file processing.
"""

import streamlit as st
import pandas as pd
import os
from main import TableExtractionOrchestrator
from budget_extractor import get_budget_extractor


def main():
    """Main application function."""
    # Configure Streamlit page
    st.set_page_config(
        page_title="Budget Table Extractor",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("Budget Table Extractor")
    st.info("This application now only supports processing files from Azure Blob Storage. Please use the main application (main.py) to process files from Azure Blob Storage.")
    st.markdown("### How to use:")
    st.markdown("1. Upload your files to Azure Blob Storage")
    st.markdown("2. Run the main application: `streamlit run main.py`")
    st.markdown("3. Select your file from the Blob Storage interface")
    
    st.warning("Local file upload has been disabled for security reasons.")


if __name__ == "__main__":
    main()
