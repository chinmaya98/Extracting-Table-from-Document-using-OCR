"""
Consolidated UI Application for Budget Table Extractor
Handles all Streamlit UI components and interactions.
"""
import streamlit as st
import pandas as pd
import os
import re
from table_extractor import TableExtractor, BlobManager
from budget_extractor import BudgetExtractor
from utils.currency_utils import contains_money


class UIApp:
    """Main UI application class."""
    
    def __init__(self):
        self.table_extractor = None
        self.blob_manager = None
        self.budget_extractor = None
        
    def initialize_app(self):
        """Initialize Streamlit app configuration."""
        st.set_page_config(page_title="Budget Table Extractor", layout="wide")
        st.title("Budget Table Extractor")
        
        # Try to initialize Azure services
        try:
            self.table_extractor = TableExtractor.get_extractor()
            self.blob_manager = TableExtractor.get_blob_manager()
            self.budget_extractor = BudgetExtractor()
            return True
        except Exception as e:
            st.error(
                "Azure configuration error. Please set your environment variables:\n"
                "- DOC_INTELLIGENCE_ENDPOINT\n"
                "- DOC_INTELLIGENCE_KEY\n" 
                "- AZURE_BLOB_CONNECTION_STRING\n"
                "- AZURE_BLOB_CONTAINER\n\n"
                f"Details: {e}"
            )
            return False

    def handle_blob_interaction(self):
        """Handle blob storage file selection."""
        if not self.blob_manager:
            return None
            
        try:
            blob_files = self.blob_manager.list_files()
            if blob_files:
                selected_file = st.selectbox("Select a file from Blob Storage", blob_files)
                return selected_file
            else:
                st.warning("No supported files found in Blob Storage.")
                return None
        except Exception as e:
            st.error(f"Error accessing Blob Storage: {e}")
            return None

    def add_download_buttons(self, df: pd.DataFrame, label_prefix: str, index: int = None):
        """Add CSV and JSON download buttons."""
        suffix = f"_{index}" if index is not None else ""
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"Download {label_prefix}{suffix} as CSV",
                data=csv_data,
                file_name=f"{label_prefix.lower()}{suffix}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            json_data = df.to_json(orient="records", indent=2).encode("utf-8")
            st.download_button(
                label=f"Download {label_prefix}{suffix} as JSON",
                data=json_data,
                file_name=f"{label_prefix.lower()}{suffix}.json",
                mime="application/json",
                use_container_width=True
            )

    def display_extraction_mode_selector(self):
        """Display mode selection UI."""
        st.markdown("### Processing Mode")
        
        mode = st.radio(
            "Choose extraction mode:",
            ["Table Extraction", "Budget Extraction"],
            index=0,
            horizontal=True
        )
            
        return mode

    def process_tables(self, file_bytes: bytes, file_name: str, mode: str):
        """Process file and extract tables based on mode."""
        file_ext = os.path.splitext(file_name)[1].lower()
        
        try:
            # Extract tables using table extractor
            if file_ext == ".pdf":
                tables = self.table_extractor.extract_from_pdf(file_bytes)
            elif file_ext in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
                tables = self.table_extractor.extract_from_image(file_bytes)
            elif file_ext in [".xlsx", ".xls"]:
                tables, sheets = self.table_extractor.extract_from_excel(file_bytes, file_name)
                return self._display_excel_results(sheets, mode)
            else:
                st.error(f"Unsupported file format: {file_ext}")
                return
            
            if not tables:
                st.warning("No tables found in the document.")
                return
            
            # Process based on mode
            if mode == "Budget Extraction":
                self._display_budget_results(tables, file_name)
            else:
                self._display_table_results(tables)
                
        except Exception as e:
            st.error(f"Error processing file: {e}")

    def _display_table_results(self, tables: list):
        """Display table extraction results."""
        st.success(f"Found {len(tables)} table(s)")
        
        all_tables = []
        
        for i, df in enumerate(tables, 1):
            if df.empty:
                continue
                
            st.subheader(f"Table {i}")
            
            # Clean and display table
            df_clean = self.table_extractor.clean_table(df)
            st.dataframe(df_clean, use_container_width=True)
            
            # Add download buttons
            self.add_download_buttons(df_clean, "Table", i)
            all_tables.append(df_clean)
            
            st.markdown("---")
        
        # Combined download
        if all_tables:
            combined_df = pd.concat(all_tables, ignore_index=True)
            st.subheader("Combined Tables")
            self.add_download_buttons(combined_df, "All_Tables")

    def _display_budget_results(self, tables: list, file_name: str):
        """Display budget extraction results."""
        budget_tables = []
        
        for i, df in enumerate(tables, 1):
            if df.empty:
                continue
                
            # Check if table contains monetary data
            if not contains_money(df):
                continue
            
            # Process for budget extraction (3-column focus)
            budget_df = self._extract_budget_columns(df)
            
            if not budget_df.empty:
                st.subheader(f"Budget Table {i}")
                st.dataframe(budget_df, use_container_width=True)
                self.add_download_buttons(budget_df, "Budget", i)
                budget_tables.append(budget_df)
                st.markdown("---")
        
        if not budget_tables:
            st.warning("No budget-related tables found.")
        else:
            # Combined budget download
            combined_budget = pd.concat(budget_tables, ignore_index=True)
            st.subheader("Combined Budget Data")
            self.add_download_buttons(combined_budget, "All_Budget_Tables")

    def _extract_budget_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract budget columns in 3-column layout: Label1, Label2, Budget_Amount."""
        if df.empty:
            return df
            
        # Use a more robust approach for 3-column standardization
        result_df = pd.DataFrame()
        
        # Find the main description/label column
        label1_col = self._find_best_description_column(df)
        
        # Find the amount/budget column  
        amount_col = self._find_best_amount_column(df)
        
        # Find secondary label/details column
        label2_col = self._find_best_secondary_column(df, exclude=[label1_col, amount_col])
        
        # Build the standardized 3-column DataFrame
        if label1_col:
            result_df['Label1'] = df[label1_col].astype(str).str.strip()
        else:
            result_df['Label1'] = 'Budget Item'
            
        if label2_col:
            result_df['Label2'] = df[label2_col].astype(str).str.strip()
        else:
            # Generate secondary labels from available data
            other_cols = [col for col in df.columns if col not in [label1_col, amount_col]]
            if other_cols:
                result_df['Label2'] = df[other_cols[0]].astype(str).str.strip()
            else:
                result_df['Label2'] = f'Item {df.index + 1}'
        
        if amount_col:
            # Extract numeric values from amount column
            result_df['Budget_Amount'] = self._extract_numeric_values(df[amount_col])
        else:
            result_df['Budget_Amount'] = 0.0
            
        # Clean up the result
        result_df = result_df.fillna('')
        result_df = result_df[result_df['Label1'].str.len() > 0]  # Remove empty labels
        result_df = result_df[result_df['Budget_Amount'] > 0]     # Remove zero amounts
        
        # Rename columns for consistency
        result_df.columns = ['Description', 'Category', 'Amount']
        
        return result_df

    def _find_best_description_column(self, df: pd.DataFrame):
        """Find the best column to use as primary description."""
        desc_keywords = ["desc", "description", "item", "name", "service", "product", 
                        "details", "category", "title", "label", "expense", "account"]
        
        # Look for keyword matches in column names
        for col in df.columns:
            col_lower = str(col).lower()
            for keyword in desc_keywords:
                if keyword in col_lower:
                    return col
                    
        # Fallback: find column with most text content
        text_scores = {}
        for col in df.columns:
            try:
                text_lengths = df[col].astype(str).str.len()
                avg_length = text_lengths.mean()
                if 3 <= avg_length <= 50:  # Reasonable text length
                    text_scores[col] = avg_length
            except:
                continue
                
        if text_scores:
            return max(text_scores.items(), key=lambda x: x[1])[0]
            
        return df.columns[0] if len(df.columns) > 0 else None

    def _find_best_amount_column(self, df: pd.DataFrame):
        """Find the best column to use as budget amount."""
        amount_keywords = ["amount", "budget", "cost", "price", "total", "value", 
                          "$", "₹", "€", "£", "usd", "inr", "eur", "gbp"]
        
        best_col = None
        max_numeric_sum = 0
        
        for col in df.columns:
            col_lower = str(col).lower()
            has_money_keyword = any(keyword in col_lower for keyword in amount_keywords)
            
            try:
                numeric_vals = self._extract_numeric_values(df[col])
                numeric_sum = numeric_vals.sum()
                
                if has_money_keyword and numeric_sum > max_numeric_sum:
                    max_numeric_sum = numeric_sum
                    best_col = col
                elif not best_col and numeric_sum > max_numeric_sum:
                    max_numeric_sum = numeric_sum
                    best_col = col
            except:
                continue
                
        return best_col

    def _find_best_secondary_column(self, df: pd.DataFrame, exclude: list):
        """Find the best column to use as secondary label."""
        exclude = [col for col in exclude if col is not None]
        
        secondary_keywords = ["qty", "quantity", "unit", "note", "remark", "detail", 
                             "type", "category", "department", "code"]
        
        # Look for columns with secondary keywords
        for col in df.columns:
            if col in exclude:
                continue
            col_lower = str(col).lower()
            for keyword in secondary_keywords:
                if keyword in col_lower:
                    return col
        
        # Fallback: return first available column not in exclude list
        for col in df.columns:
            if col not in exclude:
                return col
                
        return None

    def _extract_numeric_values(self, series: pd.Series):
        """Extract numeric values from a pandas Series."""
        def extract_number(value):
            if pd.isna(value):
                return 0.0
            
            value_str = str(value).strip()
            if not value_str:
                return 0.0
            
            # Remove currency symbols and clean up
            import re
            cleaned = re.sub(r'[^\d.,\-+]', '', value_str)
            
            # Handle different number formats
            if ',' in cleaned and '.' in cleaned:
                if cleaned.rfind(',') > cleaned.rfind('.'):
                    # European format: 1.234,56
                    cleaned = cleaned.replace('.', '').replace(',', '.')
                else:
                    # US format: 1,234.56
                    cleaned = cleaned.replace(',', '')
            elif ',' in cleaned:
                # Check if comma is likely a decimal separator
                parts = cleaned.split(',')
                if len(parts) == 2 and len(parts[1]) <= 2:
                    cleaned = cleaned.replace(',', '.')
                else:
                    cleaned = cleaned.replace(',', '')
            
            try:
                return float(cleaned)
            except (ValueError, TypeError):
                return 0.0
                
        return series.apply(extract_number)

    def _display_excel_results(self, sheets: dict, mode: str):
        """Display Excel file results."""
        st.success(f"Found {len(sheets)} sheet(s)")
        
        all_sheets = []
        
        for sheet_name, df in sheets.items():
            if df.empty:
                continue
                
            st.subheader(f"Sheet: {sheet_name}")
            
            if mode == "Budget Extraction" and contains_money(df):
                df_display = self._extract_budget_columns(df)
            else:
                df_display = df
            
            st.dataframe(df_display, use_container_width=True)
            self.add_download_buttons(df_display, f"Sheet_{sheet_name}")
            all_sheets.append(df_display)
            
            st.markdown("---")
        
        # Combined download
        if all_sheets:
            combined_df = pd.concat(all_sheets, ignore_index=True)
            st.subheader("Combined Sheets")
            self.add_download_buttons(combined_df, "All_Sheets")

    def run(self):
        """Main application runner."""
        # Initialize app
        services_available = self.initialize_app()
        
        if not services_available:
            return
        
        # Mode selection
        extraction_mode = self.display_extraction_mode_selector()
        
        # File selection
        selected_file = self.handle_blob_interaction()
        
        if selected_file:
            # Process button
            if st.button("Extract Tables", type="primary", use_container_width=True):
                with st.spinner("Processing document..."):
                    try:
                        file_bytes = self.blob_manager.download_file(selected_file)
                        self.process_tables(file_bytes, selected_file, extraction_mode)
                    except Exception as e:
                        st.error(f"Error downloading file: {e}")


# Create global app instance
app = UIApp()

def run_app():
    """Run the Streamlit application."""
    app.run()
