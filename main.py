import streamlit as st
import pandas as pd
import os
import json
import re
from dotenv import load_dotenv
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

# Load environment variables
load_dotenv()
DOC_INTELLIGENCE_ENDPOINT = os.getenv("DOC_INTELLIGENCE_ENDPOINT")
DOC_INTELLIGENCE_KEY = os.getenv("DOC_INTELLIGENCE_KEY")

st.set_page_config(layout="centered", page_title="Table Extractor with OCR")
st.title("Extract Tables with Values from PDF using Azure Document Intelligence OCR")

if not DOC_INTELLIGENCE_ENDPOINT or not DOC_INTELLIGENCE_KEY:
    st.error("Missing Azure endpoint or key in .env file")
    st.stop()

uploaded_file = st.file_uploader("Upload PDF document", type=["pdf"])

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
        max_row = max(cell.row_index for cell in table.cells)
        max_col = max(cell.column_index for cell in table.cells)

        grid = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
        
        for cell in table.cells:
            grid[cell.row_index][cell.column_index] = cell.content
        
        df = pd.DataFrame(grid)
        tables.append(df)
    
    return tables

def contains_money(df):
    money_pattern = re.compile(
        r"(\$\s?[\d,.]+(\.\d{1,2})?)|([\d,.]+\s?(usd|eur|gbp|\$|£|€))",
        re.IGNORECASE
    )
    money_keywords = ["amount", "budget", "price", "cost", "total", "value"]
    cols = [str(c).lower() for c in df.columns]

    # Check if any column name suggests money data
    if any(any(keyword in col for keyword in money_keywords) for col in cols):
        return True
    
    # Check all cell content for money pattern
    for col in df.columns:
        for cell in df[col].astype(str):
            if money_pattern.search(cell):
                return True
    
    return False

# Initialize session state variables
if "filtered_tables" not in st.session_state:
    st.session_state["filtered_tables"] = None
if "uploaded_file_name" not in st.session_state:
    st.session_state["uploaded_file_name"] = None

if uploaded_file:
    # Clear tables if user uploads a new file
    if uploaded_file.name != st.session_state["uploaded_file_name"]:
        st.session_state["uploaded_file_name"] = uploaded_file.name
        st.session_state["filtered_tables"] = None

if uploaded_file and st.button("Extract Tables from Documents"):
    with st.spinner("Extracting tables..."):
        try:
            pdf_bytes = uploaded_file.read()
            tables = extract_tables_from_pdf_ocr(pdf_bytes)
            filtered_tables = [df for df in tables if contains_money(df)]
            st.session_state["filtered_tables"] = filtered_tables

            if not filtered_tables:
                st.warning("No tables containing money or amounts found.")

        except Exception as e:
            st.error(f"Error: {e}")

# Display tables stored in session state (persist across reruns)
if st.session_state["filtered_tables"]:
    for i, df in enumerate(st.session_state["filtered_tables"]):
        st.subheader(f"Table {i+1}")
        st.dataframe(df)

    combined_csv_df = pd.concat(st.session_state["filtered_tables"], ignore_index=True)
    combined_json = [df.to_dict(orient="records") for df in st.session_state["filtered_tables"]]

    csv_data = combined_csv_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download All Tables Combined as CSV",
        data=csv_data,
        file_name="tables_combined.csv",
        mime="text/csv"
    )

    json_data = json.dumps(combined_json, indent=2).encode("utf-8")
    st.download_button(
        label="Download All Tables Combined as JSON",
        data=json_data,
        file_name="table_combined.json",
        mime="application/json"
    )

