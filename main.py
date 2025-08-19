import streamlit as st
from table_utils import AzureConfig, BlobManager, TableExtractor
import ui_handler
from budget_extractor import handle_budget_extraction, BudgetExtractor

def main():
    config = AzureConfig()
    blob_service_client = config.get_blob_service_client()
    blob_manager = BlobManager(blob_service_client, config.blob_container)
    document_client = config.get_document_client()
    extractor = TableExtractor(document_client)

    ui_handler.initialize_app()

    # Add processing mode selection at the top of the page
    st.markdown("### 🎯 Processing Mode Selection")
    
    # Create toggle buttons for processing mode selection
    col1, col2, col3 = st.columns([1.2, 1.2, 1.6])
    
    # Initialize session state for processing mode
    if 'processing_mode' not in st.session_state:
        st.session_state.processing_mode = "Table Extraction"
    
    with col1:
        # Style the button based on current selection
        button_type = "primary" if st.session_state.processing_mode == "Table Extraction" else "secondary"
        table_extraction = st.button(
            "📊 Table Extraction", 
            key="table_mode",
            help="Extract and display tables from documents",
            use_container_width=True,
            type=button_type
        )
    
    with col2:
        # Style the button based on current selection
        button_type = "primary" if st.session_state.processing_mode == "Budget Extraction" else "secondary"
        budget_extraction = st.button(
            "💰 Budget Extraction", 
            key="budget_mode",
            help="Extract budget/financial data from documents",
            use_container_width=True,
            type=button_type
        )
    
    # Update processing mode based on button clicks
    if table_extraction:
        st.session_state.processing_mode = "Table Extraction"
        st.rerun()
    elif budget_extraction:
        st.session_state.processing_mode = "Budget Extraction"
        st.rerun()
    
    st.markdown("---")

    selected_blob_file = ui_handler.handle_blob_interaction(blob_manager)

    if selected_blob_file:
        if st.session_state.processing_mode == "Table Extraction":
            # Original table extraction functionality
            ui_handler.extract_and_display_tables(blob_manager, extractor, selected_blob_file)
        
        if st.session_state.processing_mode == "Budget Extraction":
            # Budget extraction functionality - pass extractor for table extraction first
            handle_budget_extraction(selected_blob_file, blob_manager, extractor)

if __name__ == "__main__":
    main()
