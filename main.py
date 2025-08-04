import streamlit as st
import pandas as pd
import os
import io
from dotenv import load_dotenv
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient

from utils.currency_utils import contains_money  # I'll provide this next

load_dotenv()

# Azure Document Intelligence config
endpoint = os.getenv("DOC_INTELLIGENCE_ENDPOINT")
key = os.getenv("DOC_INTELLIGENCE_KEY")

# Azure Blob Storage config
blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
blob_container = os.getenv("AZURE_BLOB_CONTAINER")

client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

st.set_page_config(page_title="Budget Table Extractor", layout="wide")
st.title("📄 Budget Table Extractor")

uploaded_file = st.file_uploader("Upload a PDF or Excel file", type=["pdf", "xlsx", "xls"])

blob_files = []
selected_blob_file = None

if blob_connection_string and blob_container:
    try:
        blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
        container_client = blob_service_client.get_container_client(blob_container)
        blob_files = [b.name for b in container_client.list_blobs() if b.name.lower().endswith(('.pdf', '.xlsx', '.xls'))]
        if blob_files:
            selected_blob_file = st.selectbox("Or select a file from Blob Storage", blob_files)
    except Exception as e:
        st.error(f"Error accessing Blob Storage: {e}")

if "filtered_tables" not in st.session_state:
    st.session_state["filtered_tables"] = []

def extract_tables_from_pdf(pdf_bytes):
    poller = client.begin_analyze_document(
        model_id="prebuilt-layout",
        body=pdf_bytes,
        content_type="application/pdf"
    )
    result = poller.result()
    tables = []
    for table in result.tables:
        nrows = table.row_count
        ncols = table.column_count
        cells = [["" for _ in range(ncols)] for _ in range(nrows)]
        for cell in table.cells:
            cells[cell.row_index][cell.column_index] = cell.content
        df = pd.DataFrame(cells)
        tables.append(df)
    return tables

def extract_tables_from_excel(file_bytes):
    excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
    tables = []
    for sheet in excel_file.sheet_names:
        df = excel_file.parse(sheet)
        if not df.empty:
            tables.append(df)
    return tables

def process_file(file_bytes, file_ext):
    if file_ext == ".pdf":
        return extract_tables_from_pdf(file_bytes)
    elif file_ext in [".xlsx", ".xls"]:
        return extract_tables_from_excel(file_bytes)
    else:
        return []

if uploaded_file and st.button("Extract Tables from Uploaded File"):
    try:
        file_bytes = uploaded_file.read()
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        tables = process_file(file_bytes, ext)
        filtered_tables = [df for df in tables if contains_money(df)]
        st.session_state["filtered_tables"] = filtered_tables

        if not filtered_tables:
            st.warning("⚠️ No budget-related tables found in the uploaded document.")
    except Exception as e:
        st.error(f"❌ Error during file processing: {e}")

if selected_blob_file and st.button("Extract Tables from Blob File"):
    try:
        blob_client = blob_service_client.get_blob_client(container=blob_container, blob=selected_blob_file)
        blob_bytes = blob_client.download_blob().readall()
        ext = os.path.splitext(selected_blob_file)[1].lower()
        tables = process_file(blob_bytes, ext)
        filtered_tables = [df for df in tables if contains_money(df)]
        st.session_state["filtered_tables"] = filtered_tables

        if not filtered_tables:
            st.warning("⚠️ No budget-related tables found in the selected Blob file.")
    except Exception as e:
        st.error(f"❌ Error processing Blob file: {e}")

if st.session_state["filtered_tables"]:
    for i, df in enumerate(st.session_state["filtered_tables"]):
        st.subheader(f"📊 Table {i+1}")
        st.dataframe(df)

    combined_csv = pd.concat(st.session_state["filtered_tables"], ignore_index=True).to_csv(index=False).encode("utf-8")
    combined_json = pd.concat(st.session_state["filtered_tables"], ignore_index=True).to_json(orient="records", indent=2).encode("utf-8")

    st.download_button("⬇️ Download All Tables as CSV", combined_csv, "budget_tables.csv", "text/csv")
    st.download_button("⬇️ Download All Tables as JSON", combined_json, "budget_tables.json", "application/json")
