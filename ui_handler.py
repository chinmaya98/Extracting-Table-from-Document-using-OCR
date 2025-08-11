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

            if ext == ".pdf":
                tables = extractor.extract_from_pdf(blob_bytes)
                filtered_tables = extractor.filter_tables(tables)
                st.session_state["filtered_tables"] = filtered_tables
                st.session_state["excel_sheets"] = {}

                if not filtered_tables:
                    st.warning("⚠️ No tables found in the selected PDF file.")

            elif ext in [".xlsx", ".xls"]:
                _, _, _, sheets = read_csv_or_excel_file(blob_bytes, selected_blob_file)
                st.session_state["excel_sheets"] = sheets
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

    # Show extracted tables for PDF
    if ext == ".pdf" and st.session_state.get("filtered_tables"):
        all_cleaned_tables = []
        for i, df in enumerate(st.session_state["filtered_tables"], start=1):
            df = extractor.clean_table(df).fillna("")
            df.columns = deduplicate_columns(df.columns)
            all_cleaned_tables.append(df)

            st.subheader(f"📊 Table {i}")
            st.dataframe(df)

            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"⬇️ Download Table {i} as CSV",
                csv_data,
                f"table_{i}.csv",
                "text/csv"
            )

        combined_df = pd.concat(all_cleaned_tables, ignore_index=True)
        combined_csv = combined_df.to_csv(index=False).encode("utf-8")
        combined_json = combined_df.to_json(orient="records", indent=2).encode("utf-8")

        st.download_button("⬇️ Download All Tables as CSV", combined_csv, "budget_tables.csv", "text/csv")
        st.download_button("⬇️ Download All Tables as JSON", combined_json, "budget_tables.json", "application/json")

    # Show sheets for Excel
    if ext in [".xlsx", ".xls"] and st.session_state.get("excel_sheets"):
        sheets = st.session_state["excel_sheets"]
        all_cleaned_tables = []
        for sheet_name, sheet_df in sheets.items():
            if sheet_df.empty:
                st.warning(f"Sheet '{sheet_name}' is empty and will not be displayed.")
                continue

            sheet_df.columns = deduplicate_columns(sheet_df.columns)
            all_cleaned_tables.append(sheet_df)

            st.subheader(f"📊 Sheet: {sheet_name}")
            st.dataframe(sheet_df, use_container_width=True)

            csv_data = sheet_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"Download {sheet_name} as CSV",
                data=csv_data,
                file_name=f"{selected_blob_file}_{sheet_name}.csv",
                mime="text/csv"
            )

        combined_df = pd.concat(all_cleaned_tables, ignore_index=True)
        combined_csv = combined_df.to_csv(index=False).encode("utf-8")
        combined_json = combined_df.to_json(orient="records", indent=2).encode("utf-8")

        st.download_button("⬇️ Download All Sheets as CSV", combined_csv, "budget_sheets.csv", "text/csv")
        st.download_button("⬇️ Download All Sheets as JSON", combined_json, "budget_sheets.json", "application/json")

    return ext
