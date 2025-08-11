import streamlit as st
import pandas as pd
import os
from table_utils import AzureConfig, BlobManager, TableExtractor, read_csv_or_excel_file

def initialize_app():
    st.set_page_config(page_title="Budget Table Extractor", layout="wide")
    st.title("📄 Budget Table Extractor")

def handle_blob_interaction(blob_manager):
    blob_files = []
    selected_blob_file = None
    if blob_manager:
        try:
            blob_files = blob_manager.list_files()
            if blob_files:
                selected_blob_file = st.selectbox("Select a file from Blob Storage", blob_files)
        except Exception as e:
            st.error(f"Error accessing Blob Storage: {e}")
    return selected_blob_file

def extract_and_display_tables(blob_manager, extractor, selected_blob_file):
    if "filtered_tables" not in st.session_state:
        st.session_state["filtered_tables"] = []

    ext = None
    if selected_blob_file and st.button("Extract Tables from Selected Blob File"):
        try:
            blob_bytes = blob_manager.download_file(selected_blob_file)
            ext = os.path.splitext(selected_blob_file)[1].lower()
            if ext == ".pdf":
                tables = extractor.extract_from_pdf(blob_bytes)
            elif ext in [".xlsx", ".xls"]:
                tables_with_names, _, _, _ = read_csv_or_excel_file(blob_bytes, selected_blob_file)
                tables = [df for _, df in tables_with_names]  # Extract only DataFrames
            else:
                tables = []

            filtered_tables = extractor.filter_tables(tables)
            st.session_state["filtered_tables"] = filtered_tables
            if not filtered_tables:
                st.warning("⚠️ No budget-related tables found in the selected Blob file.")
        except Exception as e:
            st.error(f"❌ Error processing Blob file: {e}")

    if st.session_state["filtered_tables"]:
        for i, df in enumerate(st.session_state["filtered_tables"]):
            df = extractor.clean_table(df)
            df = df.fillna("")

            styled_table = df.to_html(index=False, classes="styled-table")

            st.subheader(f"📊 Table {i+1}")
            st.markdown(styled_table, unsafe_allow_html=True)

            combined_csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Table as CSV", combined_csv, f"table_{i+1}.csv", "text/csv")

        combined_csv = pd.concat(st.session_state["filtered_tables"], ignore_index=True).to_csv(index=False).encode("utf-8")
        combined_json = pd.concat(st.session_state["filtered_tables"], ignore_index=True).to_json(orient="records", indent=2).encode("utf-8")
        st.download_button("⬇️ Download All Tables as CSV", combined_csv, "budget_tables.csv", "text/csv")
        st.download_button("⬇️ Download All Tables as JSON", combined_json, "budget_tables.json", "application/json")
    return ext

def process_excel_sheets(blob_bytes, selected_blob_file):
    pass
