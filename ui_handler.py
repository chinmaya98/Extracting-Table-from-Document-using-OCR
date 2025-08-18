import streamlit as st
import pandas as pd
import os
import re
from table_utils import BlobManager, TableExtractor, read_csv_or_excel_file

def initialize_app():
    st.set_page_config(page_title="Budget Table Extractor", layout="wide")
    st.title("📄 Budget Table Extractor")

def handle_blob_interaction(blob_manager):
    blob_files = []
    selected_blob_file = None
    if blob_manager:
        try:
            # Include image files in the list
            blob_files = blob_manager.list_files(extensions=(".pdf", ".xlsx", ".xls", ".jpg", ".jpeg", ".png", ".tiff", ".bmp"))
            if blob_files:
                # Group files by type for better organization
                pdf_files = [f for f in blob_files if f.lower().endswith(('.pdf',))]
                excel_files = [f for f in blob_files if f.lower().endswith(('.xlsx', '.xls'))]
                image_files = [f for f in blob_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp'))]
                
                # Display file counts
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("PDF Files", len(pdf_files))
                with col2:
                    st.metric("Excel Files", len(excel_files))
                with col3:
                    st.metric("Image Files", len(image_files))
                
                selected_blob_file = st.selectbox(
                    "Select a file from Blob Storage", 
                    blob_files,
                    help="Supports PDF, Excel (xlsx/xls), and Image files (jpg, png, tiff, bmp)"
                )
        except Exception as e:
            st.error(f"Error accessing Blob Storage: {e}")
    return selected_blob_file

def deduplicate_columns(columns):
    seen = {}
    new_cols = []

    for col in columns:
        col_str = str(col).strip()
        if col_str.startswith("_") and col_str[1:].isdigit():
            clean_name = col_str[1:]
        else:
            clean_name = re.sub(r"#\s+(\d+)", r"#\1", col_str)
            clean_name = re.sub(r"\s+", " ", clean_name)
            clean_name = " ".join(w if w.isupper() else w.capitalize() for w in clean_name.split())
        if clean_name not in seen:
            seen[clean_name] = 0
        else:
            seen[clean_name] += 1
            clean_name = f"{clean_name}{seen[clean_name]}"
        new_cols.append(clean_name)
    return new_cols

def add_download_buttons(df: pd.DataFrame, label_prefix: str, index: int = None):
    """
    Adds CSV and JSON download buttons for a given DataFrame.
    
    Parameters:
    - df: The pandas DataFrame to download.
    - label_prefix: A prefix for button labels like 'Table' or 'Sheet'.
    - index: Optional index to add to the filename and labels (e.g. 1, 2, 3).
    """
    suffix = f"_{index}" if index is not None else ""
    csv_data = df.to_csv(index=False).encode("utf-8")
    json_data = df.to_json(orient="records", indent=2).encode("utf-8")

    st.download_button(
        label=f"⬇️ Download {label_prefix}{suffix} as CSV",
        data=csv_data,
        file_name=f"{label_prefix.lower()}{suffix}.csv",
        mime="text/csv"
    )
    st.download_button(
        label=f"⬇️ Download {label_prefix}{suffix} as JSON",
        data=json_data,
        file_name=f"{label_prefix.lower()}{suffix}.json",
        mime="application/json"
    )

def download_table_as_json(df, table_index):
    """Download a DataFrame as a JSON file."""
    json_data = df.to_json(orient="records", indent=2).encode("utf-8")
    st.download_button(
        label=f"⬇️ Download Table {table_index} as JSON",
        data=json_data,
        file_name=f"table_{table_index}.json",
        mime="application/json"
    )

def standardize_dataframe(df):
    """Ensure all columns in the DataFrame have consistent data types and flatten MultiIndex columns."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]

    df.columns = [str(col) for col in df.columns]

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
    return df

def preprocess_dataframe(df):
    """Preprocess the DataFrame by cleaning column names and ensuring consistent data types."""
    df.columns = deduplicate_columns(df.columns)
    df = standardize_dataframe(df)
    return df

def contains_money(df):
    """Check if the DataFrame contains any budget-related data, such as monetary values."""
    for col in df.select_dtypes(include=['float64', 'int64', 'object']):
        if df[col].apply(lambda x: isinstance(x, (int, float)) or (isinstance(x, str) and re.search(r'\$|€|£|¥', x))).any():
            return True
    return False

def extract_and_display_tables(blob_manager, extractor, selected_blob_file):
    if "processed_file" not in st.session_state or st.session_state["processed_file"] != selected_blob_file:
        st.session_state["processed_file"] = selected_blob_file
        st.session_state["filtered_tables"] = []
        st.session_state["excel_sheets"] = {}
        st.session_state["processed"] = False

    ext = os.path.splitext(selected_blob_file)[1].lower()

    if st.button("Extract table from Files") and not st.session_state["processed"]:
        try:
            blob_bytes = blob_manager.download_file(selected_blob_file)

            # Debugging and validation for extracted tables
            if ext == ".pdf":
                tables = extractor.extract_from_pdf(blob_bytes)
                for i, table in enumerate(tables):
                    if not isinstance(table, pd.DataFrame):
                        continue

                    if table.empty:
                        continue

                    try:
                        # Preprocess the DataFrame
                        table = preprocess_dataframe(table)

                        # Filter tables to include only budget-related ones
                        if not contains_money(table):
                            continue

                        table = standardize_dataframe(table)
                        st.session_state["filtered_tables"].append(table)
                    except Exception as e:
                        st.error(f"Error processing table {i+1}: {e}")

                if not st.session_state["filtered_tables"]:
                    st.warning("⚠️ No valid budget-related tables found in the selected PDF file.")

            elif ext in [".xlsx", ".xls"]:
                _, _, _, sheets = read_csv_or_excel_file(blob_bytes, selected_blob_file)
                st.session_state["excel_sheets"] = {sheet_name: standardize_dataframe(sheet_df) for sheet_name, sheet_df in sheets.items()}
                st.session_state["filtered_tables"] = {}

                if not sheets:
                    st.warning("⚠️ No sheets found in the Excel file.")

            else:
                st.info("Unsupported file format. Please select a PDF or Excel file.")
                st.session_state["filtered_tables"] = []
                st.session_state["excel_sheets"] = {}

            st.session_state["processed"] = True

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

    if ext == ".pdf" and st.session_state.get("filtered_tables"):
        all_cleaned_tables = []
        for i, df in enumerate(st.session_state["filtered_tables"], start=1):
            df = standardize_dataframe(extractor.clean_table(df).fillna(""))
            df.columns = deduplicate_columns(df.columns)
            all_cleaned_tables.append(df)

            st.subheader(f"📊 Table {i}")
            st.dataframe(df)

            download_table_as_json(df, i)

        combined_df = pd.concat(all_cleaned_tables, ignore_index=True)
        add_download_buttons(combined_df, label_prefix="All_Tables")

    if ext in [".xlsx", ".xls"] and st.session_state.get("excel_sheets"):
        sheets = st.session_state["excel_sheets"]
        all_cleaned_tables = []
        for i, (sheet_name, sheet_df) in enumerate(sheets.items(), start=1):
            sheet_df = standardize_dataframe(sheet_df)
            if sheet_df.empty:
                st.warning(f"Sheet '{sheet_name}' is empty and will not be displayed.")
                continue

            sheet_df.columns = deduplicate_columns(sheet_df.columns)
            all_cleaned_tables.append(sheet_df)

            st.subheader(f"📊 Sheet: {sheet_name}")
            st.dataframe(sheet_df, use_container_width=True)

            download_table_as_json(sheet_df, i)

        combined_df = pd.concat(all_cleaned_tables, ignore_index=True)
        add_download_buttons(combined_df, label_prefix="All_Sheets")

    return ext
