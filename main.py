import streamlit as st
from table_utils import AzureConfig, BlobManager, TableExtractor
import ui_handler

def main():
    ui_handler.initialize_app()

    config = AzureConfig()
    blob_manager = None
    document_client = config.get_document_client()
    extractor = TableExtractor(document_client)
    if config.blob_connection_string and config.blob_container:
        blob_service_client = config.get_blob_service_client()
        blob_manager = BlobManager(blob_service_client, config.blob_container)

    selected_blob_file = ui_handler.handle_blob_interaction(blob_manager)

    ext = ui_handler.extract_and_display_tables(blob_manager, extractor, selected_blob_file)

    if ext in [".xlsx", ".xls"]:
        blob_bytes = blob_manager.download_file(selected_blob_file)
        ui_handler.process_excel_sheets(blob_bytes, selected_blob_file)

if __name__ == "__main__":
    main()
