import streamlit as st
import pandas as pd
import os
import json
import re
from dotenv import load_dotenv
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
import io

# Load environment variables
load_dotenv()
DOC_INTELLIGENCE_ENDPOINT = os.getenv("DOC_INTELLIGENCE_ENDPOINT")
DOC_INTELLIGENCE_KEY = os.getenv("DOC_INTELLIGENCE_KEY")
BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER")

st.set_page_config(layout="centered", page_title="Table Extractor")
st.title("Extract Budget/Amount Tables from PDF or Excel")

if not DOC_INTELLIGENCE_ENDPOINT or not DOC_INTELLIGENCE_KEY:
    st.error("Missing Azure endpoint or key in .env file")
    st.stop()

# Upload UI
uploaded_file = st.file_uploader("Upload Document (PDF or Excel)", type=["pdf", "xlsx", "xls"])

# OCR extraction for PDF
def extract_tables_from_pdf_ocr(pdf_bytes):
    client = DocumentIntelligenceClient(
        endpoint=DOC_INTELLIGENCE_ENDPOINT,
        credential=AzureKeyCredential(DOC_INTELLIGENCE_KEY)
    )
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

# Detect if table contains money
def contains_money(df):
    money_pattern = re.compile(r"(\$\s?[\d,.]+(\.\d{1,2})?)|([\d,.]+\s?(usd|eur|gbp|\$|£|€))", re.IGNORECASE)
    money_keywords = ["amount", "budget", "price", "cost", "total", "value"]
    cols = [str(c).lower() for c in df.columns]
    if any(any(k in c for k in money_keywords) for c in cols):
        return True
    for col in df.columns:
        for cell in df[col].astype(str):
            if money_pattern.search(cell):
                return True
    return False

# Initialize session state
if "filtered_tables" not in st.session_state:
    st.session_state["filtered_tables"] = None
if "uploaded_file_name" not in st.session_state:
    st.session_state["uploaded_file_name"] = None

# Extract local file
if uploaded_file:
    if uploaded_file.name != st.session_state["uploaded_file_name"]:
        st.session_state["uploaded_file_name"] = uploaded_file.name
        st.session_state["filtered_tables"] = None

if uploaded_file and st.button("Extract Tables from Uploaded File"):
    with st.spinner("Processing uploaded file..."):
        try:
            file_bytes = uploaded_file.read()
            if uploaded_file.name.lower().endswith(".pdf"):
                tables = extract_tables_from_pdf_ocr(file_bytes)
            elif uploaded_file.name.lower().endswith((".xlsx", ".xls")):
                excel_df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
                tables = list(excel_df.values())
            else:
                st.warning("Unsupported file format.")
                tables = []

            filtered_tables = [df for df in tables if contains_money(df)]
            st.session_state["filtered_tables"] = filtered_tables

            if not filtered_tables:
                st.warning("No tables containing money-related data found.")
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Extract from Blob Storage
blob_files = []
selected_blob_file = None
blob_file_bytes = None

if BLOB_CONNECTION_STRING and BLOB_CONTAINER:
    try:
        blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(BLOB_CONTAINER)
        blob_files = [b.name for b in container_client.list_blobs()
                      if b.name.lower().endswith(('.pdf', '.xlsx', '.xls'))]
        if blob_files:
            selected_blob_file = st.selectbox("Or select a file from Blob Storage", blob_files)
            if selected_blob_file and st.button("Extract Tables from Blob File"):
                with st.spinner("Downloading and extracting from Blob..."):
                    blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER, blob=selected_blob_file)
                    blob_file_bytes = blob_client.download_blob().readall()

                    if selected_blob_file.lower().endswith(".pdf"):
                        tables = extract_tables_from_pdf_ocr(blob_file_bytes)
                    elif selected_blob_file.lower().endswith((".xlsx", ".xls")):
                        excel_df = pd.read_excel(io.BytesIO(blob_file_bytes), sheet_name=None)
                        tables = list(excel_df.values())
                    else:
                        tables = []

                    filtered_tables = [df for df in tables if contains_money(df)]
                    st.session_state["filtered_tables"] = filtered_tables
                    st.session_state["uploaded_file_name"] = selected_blob_file

                    if not filtered_tables:
                        st.warning("No money-related tables found in Blob.")
    except Exception as e:
        st.error(f"Error accessing Blob Storage: {e}")
else:
    st.info("Blob Storage not configured in environment variables.")

# Display extracted tables
if st.session_state["filtered_tables"]:
    for i, df in enumerate(st.session_state["filtered_tables"]):
        st.subheader(f"Table {i+1}")
        st.dataframe(df)

    combined_csv_df = pd.concat(st.session_state["filtered_tables"], ignore_index=True)
    combined_json = [df.to_dict(orient="records") for df in st.session_state["filtered_tables"]]

    st.download_button("Download All Tables as CSV",
                       data=combined_csv_df.to_csv(index=False).encode("utf-8"),
                       file_name="tables_combined.csv",
                       mime="text/csv")

    st.download_button("Download All Tables as JSON",
                       data=json.dumps(combined_json, indent=2).encode("utf-8"),
                       file_name="tables_combined.json",
                       mime="application/json")
