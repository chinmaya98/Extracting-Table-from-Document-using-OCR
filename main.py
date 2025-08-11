import streamlit as st
from table_utils import AzureConfig, BlobManager, TableExtractor
import ui_handler

def main():
    config = AzureConfig()
    blob_service_client = config.get_blob_service_client()
    blob_manager = BlobManager(blob_service_client, config.blob_container)
    document_client = config.get_document_client()
    extractor = TableExtractor(document_client)

    ui_handler.initialize_app()

    selected_blob_file = ui_handler.handle_blob_interaction(blob_manager)

    if selected_blob_file:
        # This handles both PDF and Excel inside
        ui_handler.extract_and_display_tables(blob_manager, extractor, selected_blob_file)

if __name__ == "__main__":
    main()
