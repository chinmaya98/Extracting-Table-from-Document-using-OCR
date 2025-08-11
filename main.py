import streamlit as st
import pandas as pd
import os
from table_utils import AzureConfig, BlobManager, TableExtractor

def main():
    st.set_page_config(page_title="Budget Table Extractor", layout="wide")
    st.title("📄 Budget Table Extractor")

    config = AzureConfig()
    blob_manager = None
    extractor = TableExtractor(config.endpoint, config.key)
    if config.blob_connection_string and config.blob_container:
        blob_manager = BlobManager(config.blob_connection_string, config.blob_container)

    blob_files = []
    selected_blob_file = None
    if blob_manager:
        try:
            blob_files = blob_manager.list_files()
            if blob_files:
                selected_blob_file = st.selectbox("Select a file from Blob Storage", blob_files)
        except Exception as e:
            st.error(f"Error accessing Blob Storage: {e}")

    if "filtered_tables" not in st.session_state:
        st.session_state["filtered_tables"] = []

    if selected_blob_file and st.button("Extract Tables from Selected Blob File"):
        try:
            blob_bytes = blob_manager.download_file(selected_blob_file)
            ext = os.path.splitext(selected_blob_file)[1].lower()
            tables = extractor.process_file(blob_bytes, ext)
            filtered_tables = extractor.filter_tables(tables)
            st.session_state["filtered_tables"] = filtered_tables
            if not filtered_tables:
                st.warning("⚠️ No budget-related tables found in the selected Blob file.")
        except Exception as e:
            st.error(f"❌ Error processing Blob file: {e}")

    if st.session_state["filtered_tables"]:
        for i, df in enumerate(st.session_state["filtered_tables"]):
            df = extractor.clean_table(df)
            st.subheader(f"📊 Table {i+1}")
            html_table = df.to_html(index=False, header=False, border=1)
            st.markdown(html_table, unsafe_allow_html=True)

        combined_csv = pd.concat(st.session_state["filtered_tables"], ignore_index=True).to_csv(index=False).encode("utf-8")
        combined_json = pd.concat(st.session_state["filtered_tables"], ignore_index=True).to_json(orient="records", indent=2).encode("utf-8")
        st.download_button("⬇️ Download All Tables as CSV", combined_csv, "budget_tables.csv", "text/csv")
        st.download_button("⬇️ Download All Tables as JSON", combined_json, "budget_tables.json", "application/json")

if __name__ == "__main__":
    main()
