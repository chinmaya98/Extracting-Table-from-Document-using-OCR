import streamlit as st
import pandas as pd
import os
import re
from utils.currency_utils import contains_money

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

def display_extraction_results(result):
    """Display the results from table extraction."""
    if not result['success']:
        st.error(f"❌ Error processing file: {result.get('error', 'Unknown error')}")
        return
    
    # Display budget extraction results first if available
    budget_data = result.get('budget_data')
    if budget_data is not None and not budget_data.empty:
        st.subheader("💰 Extracted Budget Data")
        st.dataframe(budget_data, use_container_width=True)
        add_download_buttons(budget_data, "Budget_Data")
        st.divider()
    
    # Display tables
    tables_to_display = result.get('budget_tables', []) if result.get('budget_tables') else result.get('tables', [])
    
    if not tables_to_display:
        if budget_data is not None and not budget_data.empty:
            st.info("ℹ️ Budget data was extracted but no raw tables are available for display.")
        else:
            st.warning("⚠️ No tables found in the processed file.")
        return
    
    # For Excel files with sheet information
    if result.get('file_type') == 'excel_csv' and result.get('budget_tables_with_names'):
        display_excel_results(result)
    else:
        display_pdf_image_results(tables_to_display)


def display_excel_results(result):
    """Display results for Excel/CSV files with sheet information."""
    budget_tables_with_names = result.get('budget_tables_with_names', [])
    
    if not budget_tables_with_names:
        st.warning("⚠️ No budget-related tables found in Excel file.")
        return
    
    all_cleaned_tables = []
    
    for i, (sheet_name, df) in enumerate(budget_tables_with_names, start=1):
        if df.empty:
            st.warning(f"Sheet '{sheet_name}' is empty and will not be displayed.")
            continue
        
        # Clean and preprocess the DataFrame
        df = preprocess_dataframe(df)
        all_cleaned_tables.append(df)
        
        st.subheader(f"📊 Sheet: {sheet_name}")
        st.dataframe(df, use_container_width=True)
        
        # Add download buttons
        add_download_buttons(df, f"Sheet_{sheet_name}", index=i)
    
    # Combined download option
    if len(all_cleaned_tables) > 1:
        combined_df = pd.concat(all_cleaned_tables, ignore_index=True)
        st.subheader("📊 Combined Data")
        add_download_buttons(combined_df, "All_Sheets")


def display_pdf_image_results(tables):
    """Display results for PDF/Image files."""
    all_cleaned_tables = []
    
    for i, df in enumerate(tables, start=1):
        if df.empty:
            continue
        
        # Clean and preprocess the DataFrame
        df = preprocess_dataframe(df)
        all_cleaned_tables.append(df)
        
        st.subheader(f"📊 Table {i}")
        st.dataframe(df, use_container_width=True)
        
        # Add download buttons
        add_download_buttons(df, f"Table", index=i)
    
    # Combined download option
    if len(all_cleaned_tables) > 1:
        combined_df = pd.concat(all_cleaned_tables, ignore_index=True)
        st.subheader("📊 Combined Data")
        add_download_buttons(combined_df, "All_Tables")


def extract_and_display_tables(blob_manager, extractor, selected_blob_file):
    """Legacy function - kept for backward compatibility but deprecated."""
    st.warning("⚠️ This function is deprecated. Please use the new orchestrator in main.py")
    
    # Basic file processing without the full orchestrator
    ext = os.path.splitext(selected_blob_file)[1].lower()
    
    if st.button("Extract Tables"):
        try:
            blob_bytes = blob_manager.download_file(selected_blob_file)
            
            # Simple processing based on file type
            if ext == ".pdf":
                tables = extractor.extract_from_pdf(blob_bytes)
                display_simple_tables(tables, "PDF")
            elif ext in [".xlsx", ".xls"]:
                # This would need the Excel processor
                st.error("Excel processing requires the new orchestrator. Please use main.py with --mode web")
            elif ext in [".png", ".jpg", ".jpeg", ".tiff"]:
                tables = extractor.extract_from_image(blob_bytes)
                display_simple_tables(tables, "Image")
            else:
                st.error("Unsupported file format")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")


def display_simple_tables(tables, file_type):
    """Display simple table results."""
    if not tables:
        st.warning(f"⚠️ No tables found in the {file_type} file.")
        return
    
    for i, table in enumerate(tables, start=1):
        if isinstance(table, pd.DataFrame) and not table.empty:
            table = preprocess_dataframe(table)
            st.subheader(f"📊 Table {i}")
            st.dataframe(table, use_container_width=True)
            add_download_buttons(table, f"Table", index=i)
