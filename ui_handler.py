import streamlit as st
import pandas as pd
import os
import re

def initialize_app():
    """Initialize Streamlit app configuration."""
    st.set_page_config(page_title="Budget Table Extractor", layout="wide")
    st.title("Budget Table Extractor")

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
    """Clean and deduplicate column names."""
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

def standardize_dataframe(df):
    """Ensure all columns have consistent data types."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]

    df.columns = [str(col) for col in df.columns]
    
    for col in df.columns:
        df[col] = df[col].astype('object').where(df[col].notna(), '')
    
    return df

def preprocess_dataframe(df):
    """Preprocess the DataFrame by cleaning column names and ensuring consistent data types."""
    df.columns = deduplicate_columns(df.columns)
    df = standardize_dataframe(df)
    return df

def display_extraction_results(result):
    """Display the results from table extraction."""
    if not result['success']:
        st.error(f"❌ Error processing file: {result.get('error', 'Unknown error')}")
        return
    
    # Display budget extraction results first if available (HIDDEN)
    budget_data = result.get('budget_data')
    if budget_data is not None and not budget_data.empty:
        # st.subheader("💰 Extracted Budget Data")
        # st.dataframe(budget_data, use_container_width=True)
        # add_download_buttons(budget_data, "Budget_Data")
        # st.divider()
        # st.title("TABLE EXTRACTION")
        pass  # Budget data processing continues but display is hidden
    
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
    """Display all Excel/CSV sheets in a collapsible expander for each sheet."""
    all_sheets = result.get('sheets', {})
    if not all_sheets:
        st.warning("⚠️ No tables found in Excel file.")
        return

    for sheet_name, df in all_sheets.items():
        if df.empty:
            continue
        df = preprocess_dataframe(df)
        display_df = df.fillna('')
        # Convert all columns to string to avoid ArrowTypeError
        display_df = display_df.astype(str)
        with st.expander(f"{sheet_name}", expanded=True):
            st.dataframe(display_df, use_container_width=True)
            json_data = df.to_json(orient='records', indent=2).encode('utf-8')
            st.download_button(
                f"⬇️ Download '{sheet_name}' as JSON",
                data=json_data,
                file_name=f"{sheet_name.replace(' ', '_')}.json",
                mime="application/json",
                key=f"json_{sheet_name}"
            )


def display_pdf_image_results(tables):
    """Display results for PDF/Image files."""
    all_cleaned_tables = []
    for i, df in enumerate(tables, start=1):
        if df.empty:
            continue
        
        # Clean and preprocess the DataFrame
        df = preprocess_dataframe(df)
        all_cleaned_tables.append(df)
        display_df = df.fillna('').astype(str)  # Ensure Arrow compatibility
        st.subheader(f"📊 Table {i}")
        st.dataframe(display_df, use_container_width=True)
        json_data = df.to_json(orient='records', indent=2).encode('utf-8')
        st.download_button(
            f"⬇️ Download Table {i} as JSON",
            data=json_data,
            file_name=f"table_{i}.json",
            mime="application/json"
        )

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
            display_df = table.fillna('').astype(str)  # Ensure Arrow compatibility
            st.subheader(f"📊 Table {i}")
            st.dataframe(display_df, use_container_width=True)
            json_data = table.to_json(orient='records', indent=2).encode('utf-8')
            st.download_button(
                f"⬇️ Download Table {i} as JSON",
                data=json_data,
                file_name=f"table_{i}.json",
                mime="application/json"
            )
