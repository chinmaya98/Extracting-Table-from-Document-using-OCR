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

    # Handle empty/null/NaN values properly - keep them as empty strings, don't fill with text
    for col in df.columns:
        if df[col].dtype == 'object':
            # Replace various null representations with empty strings
            df[col] = df[col].replace([pd.NA, None, 'nan', 'NaN', 'NULL', 'null', 'None', 'N/A', 'n/a', '#N/A', '#NULL!'], '')
            df[col] = df[col].astype(str)
            # Clean up any remaining 'nan' strings
            df[col] = df[col].replace('nan', '')
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna('')  # Use empty string instead of 0 for display
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
        tables = st.session_state["filtered_tables"]
        all_cleaned_tables = []
        
        if len(tables) > 1:
            # Display multiple tables in 3-column layout
            st.subheader(f"📊 Extracted Tables ({len(tables)} tables found)")
            
            # Group tables into rows of 3
            tables_per_row = 3
            table_groups = []
            current_group = []
            
            for i, df in enumerate(tables, start=1):
                df = standardize_dataframe(extractor.clean_table(df))
                # Handle empty values properly - ensure they show as empty cells
                df = df.replace(['nan', 'NaN', 'NULL', 'null', 'None', 'N/A', 'n/a'], '')
                df.columns = deduplicate_columns(df.columns)
                all_cleaned_tables.append(df)
                
                current_group.append((i, df))
                
                if len(current_group) == tables_per_row or i == len(tables):
                    table_groups.append(current_group)
                    current_group = []
            
            # Display each group of tables in columns
            for group in table_groups:
                cols = st.columns(len(group))
                
                for col_idx, (table_num, table_df) in enumerate(group):
                    with cols[col_idx]:
                        st.markdown(f"**Table {table_num}**")
                        st.dataframe(table_df, use_container_width=True, height=300)
                        
                        # Individual download button
                        csv_data = table_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label=f"📥 CSV",
                            data=csv_data,
                            file_name=f"table_{table_num}.csv",
                            mime="text/csv",
                            key=f"table_csv_{table_num}"
                        )
        else:
            # Single table display
            for i, df in enumerate(tables, start=1):
                df = standardize_dataframe(extractor.clean_table(df))
                df = df.replace(['nan', 'NaN', 'NULL', 'null', 'None', 'N/A', 'n/a'], '')
                df.columns = deduplicate_columns(df.columns)
                all_cleaned_tables.append(df)

                st.subheader(f"📊 Table {i}")
                st.dataframe(df)

                download_table_as_json(df, i)

        # Combined download option
        if all_cleaned_tables:
            combined_df = pd.concat(all_cleaned_tables, ignore_index=True)
            add_download_buttons(combined_df, label_prefix="All_Tables")

    if ext in [".xlsx", ".xls"] and st.session_state.get("excel_sheets"):
        sheets = st.session_state["excel_sheets"]
        sheet_items = list(sheets.items())
        all_cleaned_tables = []
        
        if len(sheet_items) > 1:
            # Display multiple sheets in 3-column layout
            st.subheader(f"📊 Excel Sheets ({len(sheet_items)} sheets found)")
            
            # Group sheets into rows of 3
            sheets_per_row = 3
            sheet_groups = []
            current_group = []
            
            for i, (sheet_name, sheet_df) in enumerate(sheet_items, start=1):
                sheet_df = standardize_dataframe(sheet_df)
                if sheet_df.empty:
                    continue
                
                # Handle empty values properly - ensure they show as empty cells
                sheet_df = sheet_df.replace(['nan', 'NaN', 'NULL', 'null', 'None', 'N/A', 'n/a'], '')
                sheet_df.columns = deduplicate_columns(sheet_df.columns)
                all_cleaned_tables.append(sheet_df)
                
                current_group.append((sheet_name, sheet_df))
                
                if len(current_group) == sheets_per_row or i == len(sheet_items):
                    sheet_groups.append(current_group)
                    current_group = []
            
            # Display each group of sheets in columns
            for group in sheet_groups:
                cols = st.columns(len(group))
                
                for col_idx, (sheet_name, sheet_df) in enumerate(group):
                    with cols[col_idx]:
                        st.markdown(f"**Sheet: {sheet_name}**")
                        st.dataframe(sheet_df, use_container_width=True, height=300)
                        
                        # Individual download button
                        csv_data = sheet_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label=f"📥 CSV",
                            data=csv_data,
                            file_name=f"sheet_{sheet_name}.csv",
                            mime="text/csv",
                            key=f"sheet_csv_{sheet_name}"
                        )
        else:
            # Single sheet display
            for i, (sheet_name, sheet_df) in enumerate(sheet_items, start=1):
                sheet_df = standardize_dataframe(sheet_df)
                if sheet_df.empty:
                    st.warning(f"Sheet '{sheet_name}' is empty and will not be displayed.")
                    continue

                sheet_df = sheet_df.replace(['nan', 'NaN', 'NULL', 'null', 'None', 'N/A', 'n/a'], '')
                sheet_df.columns = deduplicate_columns(sheet_df.columns)
                all_cleaned_tables.append(sheet_df)

                st.subheader(f"📊 Sheet: {sheet_name}")
                st.dataframe(sheet_df, use_container_width=True)

                download_table_as_json(sheet_df, i)

        # Combined download option
        if all_cleaned_tables:
            combined_df = pd.concat(all_cleaned_tables, ignore_index=True)
            add_download_buttons(combined_df, label_prefix="All_Sheets")

    return ext
