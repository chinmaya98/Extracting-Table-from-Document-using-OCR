
import streamlit as st
import pandas as pd
import time # For simulating connection delay
import json
import os



# --- Load storage account info from tbudgetdb.json ---
json_path = "tbudgetdb.json"
account_name = ""
blob_endpoint = ""
default_container = "budget-docs"  # fallback
if os.path.exists(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
        # Try to get account name and blob endpoint
        account_name = data.get("name", "")
        blob_endpoint = data.get("primaryEndpoints", {}).get("blob", "")
        # Try to get default container name if present
        # Look for a property named 'container', else fallback
        default_container = data.get("container", default_container)

# --- Load default key from .env ---
from dotenv import load_dotenv
load_dotenv()
default_key = os.getenv("AZURE_BLOB_KEY", "")


def run_connection_and_show_data(selected_path):
    """
    Connects to Azure Blob Storage and reads a file of any format.
    """
    from azure.storage.blob import BlobServiceClient
    import io
    with st.spinner('Connecting to Azure Blob Storage... 💸'):
        time.sleep(1)
    try:
        blob_service_client = BlobServiceClient.from_connection_string(selected_path['connection_string'])
        blob_client = blob_service_client.get_blob_client(container=selected_path['container_name'], blob=selected_path['blob_name'])
        blob_data = blob_client.download_blob().readall()
        file_format = selected_path['file_format'].lower()
        # OCR logic for any file format
        import pytesseract
        from PIL import Image
        import tempfile
        import pdfplumber
        import re
        ocr_text = None
        # PDF logic: extract tables as-is, then OCR for summary if needed
        if file_format == 'pdf' or selected_path['blob_name'].lower().endswith('.pdf'):
            import numpy as np
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
                tmp_pdf.write(blob_data)
                tmp_pdf.flush()
                with pdfplumber.open(tmp_pdf.name) as pdf:
                    all_tables = []
                    for page in pdf.pages:
                        tables = page.extract_tables()
                        for table in tables:
                            # Convert to DataFrame and append if not empty
                            df_table = pd.DataFrame(table[1:], columns=table[0]) if len(table) > 1 else pd.DataFrame(table)
                            if not df_table.empty:
                                all_tables.append(df_table)
                    if all_tables:
                        def get_label_amount_table(df):
                            cols = [c.lower() for c in df.columns]
                            label_col = None
                            amount_col = None
                            for c in df.columns:
                                if 'label' in c.lower() or 'item' in c.lower() or 'service' in c.lower() or 'description' in c.lower():
                                    label_col = c
                                if 'amount' in c.lower() or 'total' in c.lower() or 'price' in c.lower() or 'value' in c.lower():
                                    amount_col = c
                            if label_col and amount_col:
                                return df[[label_col, amount_col]].rename(columns={label_col: 'Label', amount_col: 'Amount'})
                            if len(df.columns) == 2:
                                return df.rename(columns={df.columns[0]: 'Label', df.columns[1]: 'Amount'})
                            return None
                        readable_tables = []
                        extracted_tables = []
                        for i, df_table in enumerate(all_tables):
                            label_amount_df = get_label_amount_table(df_table)
                            if label_amount_df is not None:
                                readable_tables.append(label_amount_df)
                                extracted_tables.append((f"Table {i+1} (Label/Amount)", label_amount_df))
                            else:
                                extracted_tables.append((f"Table {i+1}", df_table))
                        # Only show the final generated table and download buttons (hide extracted tables and button)
                        if readable_tables:
                            df_all = pd.concat(readable_tables, ignore_index=True)
                            csv_data = df_all.to_csv(index=False).encode('utf-8')
                            json_data = df_all.to_json(orient='records', force_ascii=False, indent=2).encode('utf-8')
                            st.success("Extracted tables from PDF!")
                            st.subheader("Generated Data Table")
                            st.dataframe(df_all)
                            st.download_button(
                                label="Download table as CSV",
                                data=csv_data,
                                file_name="pdf_label_amount_tables.csv",
                                mime="text/csv"
                            )
                            st.download_button(
                                label="Download table as JSON",
                                data=json_data,
                                file_name="pdf_label_amount_tables.json",
                                mime="application/json"
                            )
                            return df_all
                        else:
                            df_all = pd.concat(all_tables, ignore_index=True)
                            csv_data = df_all.to_csv(index=False).encode('utf-8')
                            json_data = df_all.to_json(orient='records', force_ascii=False, indent=2).encode('utf-8')
                            st.success("Extracted tables from PDF!")
                            st.subheader("Generated Data Table")
                            st.dataframe(df_all)
                            st.download_button(
                                label="Download table as CSV",
                                data=csv_data,
                                file_name="pdf_tables.csv",
                                mime="text/csv"
                            )
                            st.download_button(
                                label="Download table as JSON",
                                data=json_data,
                                file_name="pdf_tables.json",
                                mime="application/json"
                            )
                            return df_all
                    # If no tables found, fallback to OCR summary extraction
                    if not all_tables:
                        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                        ocr_text = text
                        # OCR summary extraction as before
                        summary_pattern = r"(?i)(total|subtotal|budget|[A-Za-z0-9 \-]+)\s*[:\-]?\s*\$?([\d,]+(?:\.\d{2})?)"
                        summary_matches = re.findall(summary_pattern, ocr_text)
                        table_rows = []
                        label_count = {}
                        subtotal_row = None
                        total_row = None
                        address_keywords = ["pin", "house", "street", "road", "avenue", "block", "no", "number", "zip", "postal", "address"]
                        city_names = [
                            "new york", "los angeles", "chicago", "houston", "phoenix", "philadelphia", "san antonio", "san diego", "dallas", "san jose", "austin", "jacksonville", "fort worth", "columbus", "charlotte", "san francisco", "indianapolis", "seattle", "denver", "washington", "boston", "el paso", "nashville", "detroit", "oklahoma city", "portland", "las vegas", "memphis", "louisville", "baltimore", "milwaukee", "albuquerque", "tucson", "fresno", "sacramento", "kansas city", "long beach", "mesa", "atlanta", "colorado springs", "virginia beach", "raleigh", "omaha", "miami", "oakland", "minneapolis", "tulsa", "wichita", "new orleans", "arlington"
                        ]
                        for label, amount in summary_matches:
                            label_clean = label.strip()
                            if label_clean.isdigit():
                                continue
                            if all(c in ",.:-_()[]{}|/\\'\"!@#$%^&*" for c in label_clean):
                                continue
                            if all(c in ",.:-_()[]{}|/\\'\"!@#$%^&*" for c in amount.strip()):
                                continue
                            label_key = label_clean.lower()
                            if any(addr in label_key for addr in address_keywords):
                                continue
                            if label_key in city_names:
                                continue
                            if label_key in label_count:
                                label_count[label_key] += 1
                                label_display = f"{label_clean}_x{label_count[label_key]}"
                            else:
                                label_count[label_key] = 1
                                label_display = label_clean
                            if label_key in ["subtotal"]:
                                subtotal_row = {"Label": "Subtotal", "Amount": amount}
                            elif label_key in ["total", "amount", "total amount", "grand total"]:
                                total_row = {"Label": "Total", "Amount": amount}
                            else:
                                table_rows.append({"Label": label_display, "Amount": amount})
                        seen = set()
                        unique_rows = []
                        for row in table_rows:
                            key = (row["Label"].lower(), row["Amount"])
                            if key not in seen:
                                unique_rows.append(row)
                                seen.add(key)
                        if subtotal_row:
                            unique_rows.append(subtotal_row)
                        if total_row:
                            unique_rows.append(total_row)
                        if unique_rows:
                            df = pd.DataFrame(unique_rows)
                            st.success("Extracted items/services and amounts from PDF (OCR)!")
                            st.dataframe(df)
                            if total_row:
                                st.markdown(f"**Total Amount:** {total_row['Amount']}")
                            return df
                        else:
                            st.warning("No items/services or amounts found in PDF.")
                            possible_labels = ["Item/Service", "Subtotal", "Total"]
                            df = pd.DataFrame({"Label": possible_labels, "Amount": ["-"]*len(possible_labels)})
                            st.dataframe(df)
                            return df
        # Try OCR if file is image or fallback for other formats
        if file_format in ['png', 'jpg', 'jpeg', 'bmp', 'tiff'] or selected_path['blob_name'].lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.' + file_format) as tmp_img:
                tmp_img.write(blob_data)
                tmp_img.flush()
                image = Image.open(tmp_img.name)
                ocr_text = pytesseract.image_to_string(image)
        else:
            # Try to open as image anyway
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_img:
                    tmp_img.write(blob_data)
                    tmp_img.flush()
                    image = Image.open(tmp_img.name)
                    ocr_text = pytesseract.image_to_string(image)
            except Exception:
                ocr_text = None
        if ocr_text:
            # For non-PDF files, try to extract all items/services and amounts
            summary_pattern = r"(?i)(total|subtotal|budget|[A-Za-z0-9 \-]+)\s*[:\-]?\s*\$?([\d,]+(?:\.\d{2})?)"
            summary_matches = re.findall(summary_pattern, ocr_text)
            table_rows = []
            label_count = {}
            subtotal_row = None
            total_row = None
            address_keywords = ["pin", "house", "street", "road", "avenue", "block", "no", "number", "zip", "postal", "address"]
            for label, amount in summary_matches:
                label_clean = label.strip()
                if label_clean.isdigit():
                    continue
                # Ignore amounts that are only punctuation/symbols
                if all(c in ",.:-_()[]{}|/\\'\"!@#$%^&*" for c in amount.strip()):
                    continue
                label_key = label_clean.lower()
                if any(addr in label_key for addr in address_keywords):
                    continue
                if label_key in label_count:
                    label_count[label_key] += 1
                    label_display = f"{label_clean}_x{label_count[label_key]}"
                else:
                    label_count[label_key] = 1
                    label_display = label_clean
                if label_key in ["subtotal"]:
                    subtotal_row = {"Label": "Subtotal", "Amount": amount}
                elif label_key in ["total", "amount", "total amount", "grand total"]:
                    total_row = {"Label": "Total", "Amount": amount}
                else:
                    table_rows.append({"Label": label_display, "Amount": amount})
            seen = set()
            unique_rows = []
            for row in table_rows:
                key = (row["Label"].lower(), row["Amount"])
                if key not in seen:
                    unique_rows.append(row)
                    seen.add(key)
            if subtotal_row:
                unique_rows.append(subtotal_row)
            if total_row:
                unique_rows.append(total_row)
            if unique_rows:
                df = pd.DataFrame(unique_rows)
                st.success("Extracted items/services and amounts from file!")
                st.dataframe(df)
                if total_row:
                    st.markdown(f"**Total Amount:** {total_row['Amount']}")
                return df
            else:
                st.warning("No items/services or amounts found in file.")
                possible_labels = ["Item/Service", "Subtotal", "Total"]
                df = pd.DataFrame({"Label": possible_labels, "Amount": ["-"]*len(possible_labels)})
                st.dataframe(df)
                return df
        # Fallback to original logic
        if file_format == 'csv':
            df = pd.read_csv(io.BytesIO(blob_data))
            st.success(f"Successfully loaded CSV file from blob storage!")
            st.dataframe(df)
            return df
        elif file_format == 'excel' or file_format == 'xlsx':
            excel_file = pd.ExcelFile(io.BytesIO(blob_data))
            all_tables = []
            readable_tables = []
            for sheet_name in excel_file.sheet_names:
                df = excel_file.parse(sheet_name)
                if not df.empty:
                    all_tables.append(df)
                    # Try to extract label/amount columns
                    cols = [c.lower() for c in df.columns]
                    label_col = None
                    amount_col = None
                    for c in df.columns:
                        if 'label' in c.lower() or 'item' in c.lower() or 'service' in c.lower() or 'description' in c.lower():
                            label_col = c
                        if 'amount' in c.lower() or 'total' in c.lower() or 'price' in c.lower() or 'value' in c.lower():
                            amount_col = c
                    if label_col and amount_col:
                        label_amount_df = df[[label_col, amount_col]].rename(columns={label_col: 'Label', amount_col: 'Amount'})
                        st.write(f"Sheet: {sheet_name} (Label/Amount)")
                        st.dataframe(label_amount_df)
                        readable_tables.append(label_amount_df)
                    elif len(df.columns) == 2:
                        label_amount_df = df.rename(columns={df.columns[0]: 'Label', df.columns[1]: 'Amount'})
                        st.write(f"Sheet: {sheet_name} (Label/Amount)")
                        st.dataframe(label_amount_df)
                        readable_tables.append(label_amount_df)
                    else:
                        st.write(f"Sheet: {sheet_name}")
                        st.dataframe(df)
            if readable_tables:
                df_all = pd.concat(readable_tables, ignore_index=True)
                csv_data = df_all.to_csv(index=False).encode('utf-8')
                json_data = df_all.to_json(orient='records', force_ascii=False, indent=2).encode('utf-8')
                st.download_button(
                    label="Download all label/amount tables as CSV",
                    data=csv_data,
                    file_name="excel_label_amount_tables.csv",
                    mime="text/csv"
                )
                st.download_button(
                    label="Download all label/amount tables as JSON",
                    data=json_data,
                    file_name="excel_label_amount_tables.json",
                    mime="application/json"
                )
                return df_all
            elif all_tables:
                df_all = pd.concat(all_tables, ignore_index=True)
                csv_data = df_all.to_csv(index=False).encode('utf-8')
                json_data = df_all.to_json(orient='records', force_ascii=False, indent=2).encode('utf-8')
                st.download_button(
                    label="Download all tables as CSV",
                    data=csv_data,
                    file_name="excel_tables.csv",
                    mime="text/csv"
                )
                st.download_button(
                    label="Download all tables as JSON",
                    data=json_data,
                    file_name="excel_tables.json",
                    mime="application/json"
                )
                return df_all
        elif file_format == 'json':
            df = pd.read_json(io.BytesIO(blob_data))
            st.success(f"Successfully loaded JSON file from blob storage!")
            st.dataframe(df)
            return df
        else:
            st.info("File format not recognized for tabular display. Downloading raw file.")
            return blob_data
    except Exception as e:
        st.error(f"Error reading from blob storage: {e}")
        return None


st.set_page_config(layout="centered", page_title="Blob Data Viewer")

st.title("Azure Blob Storage File Reader")

# --- Azure Blob Storage Details ---



st.header("1. Azure Blob Storage Connection")
# Hide technical details for a cleaner UI
# st.write(f"**Storage Account Name:** {account_name}")
# st.write(f"**Blob Endpoint:** {blob_endpoint}")


# Use default key from .env if present, else prompt for it
if default_key:
    account_key = default_key
    # st.write("Using default key from .env file.")  # Hide technical info
else:
    account_key = st.text_input("Account Key (from Azure Portal)", type="password")


# Use container name from .env or fallback

# --- Incorporate check_blob.py logic for blob listing ---
from azure.storage.blob import BlobServiceClient

# Use container name from .env or fallback
container_name = os.getenv("AZURE_BLOB_CONTAINER", default_container)

# Build connection string dynamically
connection_string = os.getenv(
    "AZURE_BLOB_CONNECTION_STRING",
    f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
)

blob_name = None
files = []
if connection_string and container_name:
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        # Hide blob listing status for clean UI
        for blob in container_client.list_blobs():
            name = blob.name
            # Only include files in the root (no '/')
            if '/' not in name:
                files.append(name)
        if files:
            blob_name = st.selectbox("Select a file", sorted(files))
        else:
            st.warning("No files found in the root of the container.")
    except Exception as e:
        st.error("Unable to access Azure Blob Storage. Please check your credentials or try again later.")
else:
    st.warning("Unable to access blob storage. Please check your credentials.")


# Infer file format from file extension
file_format = "other"
if blob_name:
    if blob_name.lower().endswith(".csv"):
        file_format = "csv"
    elif blob_name.lower().endswith(".xlsx") or blob_name.lower().endswith(".xls"):
        file_format = "excel"
    elif blob_name.lower().endswith(".json"):
        file_format = "json"
    elif blob_name.lower().endswith(".pdf"):
        file_format = "pdf"

selected_path_value = {
    'connection_string': connection_string,
    'container_name': container_name,
    'blob_name': blob_name,
    'file_format': file_format
}

# --- RUN Button ---

# Show table when a file is selected and RUN is pressed
if blob_name and st.button("RUN"):
    selected_path_value = {
        'connection_string': connection_string,
        'container_name': container_name,
        'blob_name': blob_name,
        'file_format': file_format
    }
    run_connection_and_show_data(selected_path_value)

st.sidebar.header("About")
st.sidebar.info("This is a simple Streamlit application demonstrating path selection and data display.")

st.sidebar.header("Instructions")
st.sidebar.write("1. Select a storage path from the dropdown.")
st.sidebar.write("2. Click the 'RUN' button to simulate connecting and displaying data.")