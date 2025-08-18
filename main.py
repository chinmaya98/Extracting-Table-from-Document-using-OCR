import streamlit as st
from table_utils import AzureConfig, BlobManager, TableExtractor
import ui_handler
from budget_extractor import handle_budget_extraction

def main():
    config = AzureConfig()
    blob_service_client = config.get_blob_service_client()
    blob_manager = BlobManager(blob_service_client, config.blob_container)
    document_client = config.get_document_client()
    extractor = TableExtractor(document_client)

    ui_handler.initialize_app()

    # Add processing mode selection
    st.sidebar.title("Processing Options")
    processing_mode = st.sidebar.radio(
        "Select Processing Mode:",
        ["Table Extraction", "Budget Extraction", "Both"]
    )

    selected_blob_file = ui_handler.handle_blob_interaction(blob_manager)

    if selected_blob_file:
        if processing_mode in ["Table Extraction", "Both"]:
            # Original table extraction functionality
            ui_handler.extract_and_display_tables(blob_manager, extractor, selected_blob_file)
        
        if processing_mode in ["Budget Extraction", "Both"]:
            # Budget extraction functionality (handled in budget_extractor.py)
            handle_budget_extraction(selected_blob_file)

if __name__ == "__main__":
    main()
