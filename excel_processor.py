"""
Excel Table Extraction Module
Handles Excel files (.xlsx, .xls) and CSV files for table extraction.
"""

import io
import pandas as pd
from utils.currency_utils import contains_money


class ExcelProcessor:
    """
    Processes Excel and CSV files for table extraction.
    """
    
    def __init__(self):
        """Initialize the Excel processor."""
        pass
    
    def extract_tables_from_excel(self, file_bytes, file_name="", max_null_threshold=0.8):
        """
        Extract tables from Excel file, handling multiple sheets.
        
        Args:
            file_bytes: Excel file content as bytes
            file_name: Original filename (for error reporting)
            max_null_threshold: (ignored, kept for compatibility)
            
        Returns:
            Tuple: (tables_list, sheets_dict, metadata)
            - tables_list: List of (sheet_name, DataFrame) tuples
            - sheets_dict: Dictionary of {sheet_name: DataFrame}
            - metadata: Dictionary with extraction metadata
        """
        try:
            # Read Excel file
            excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
            
            sheets = {}
            tables = []
            metadata = {
                'total_sheets': len(excel_file.sheet_names),
                'processed_sheets': 0,
                'empty_sheets': 0,
                'sheet_info': []
            }
            
            for sheet_name in excel_file.sheet_names:
                try:
                    # Read sheet without header to get raw data
                    sheet_df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
                    
                    # Show ALL sheets, including empty ones
                    # Do not modify or fill null values - keep them as NaN/null
                    sheets[sheet_name] = sheet_df
                    tables.append((sheet_name, sheet_df))
                    metadata['processed_sheets'] += 1
                    
                    if sheet_df.empty:
                        metadata['empty_sheets'] += 1
                    
                    # Add sheet info to metadata
                    sheet_info = {
                        'sheet_name': sheet_name,
                        'rows': len(sheet_df),
                        'columns': len(sheet_df.columns),
                        'has_monetary_data': contains_money(sheet_df) if not sheet_df.empty else False
                    }
                    metadata['sheet_info'].append(sheet_info)
                except Exception as e:
                    print(f"Error processing sheet '{sheet_name}': {e}")
                    continue
            
            return tables, sheets, metadata
            
        except Exception as e:
            raise RuntimeError(f"Error reading Excel file '{file_name}': {e}")
    
    def extract_tables_from_csv(self, file_bytes, file_name="", max_null_threshold=0.8):
        """
        Extract table from CSV file.
        
        Args:
            file_bytes: CSV file content as bytes
            file_name: Original filename (for error reporting)
            max_null_threshold: (ignored, kept for compatibility)
            
        Returns:
            Tuple: (tables_list, sheets_dict, metadata)
        """
        try:
            # Read CSV file
            df = pd.read_csv(io.BytesIO(file_bytes))
            # Show CSV as-is, do not clean or modify
            metadata = {
                'total_sheets': 1,
                'processed_sheets': 1 if not df.empty else 0,
                'empty_sheets': 1 if df.empty else 0,
                'sheet_info': [{
                    'sheet_name': 'CSV',
                    'rows': len(df),
                    'columns': len(df.columns),
                    'has_monetary_data': contains_money(df)
                }] if not df.empty else []
            }
            if df.empty:
                return [], {}, metadata
            tables = [("CSV", df)]
            sheets = {"CSV": df}
            return tables, sheets, metadata
        except Exception as e:
            raise RuntimeError(f"Error reading CSV file '{file_name}': {e}")
    
    def process_file(self, file_bytes, file_extension, file_name="", max_null_threshold=0.8):
        """
        Process Excel or CSV file and extract tables.
        
        Args:
            file_bytes: File content as bytes
            file_extension: File extension (e.g., '.xlsx', '.csv')
            file_name: Original filename
            max_null_threshold: Float (0-1) indicating maximum proportion of null/empty cells allowed in a row
            
        Returns:
            Tuple: (tables_list, sheets_dict, metadata)
        """
        try:
            file_ext = file_extension.lower()
            
            if file_ext == '.csv':
                return self.extract_tables_from_csv(file_bytes, file_name, max_null_threshold)
            elif file_ext in ['.xlsx', '.xls']:
                return self.extract_tables_from_excel(file_bytes, file_name, max_null_threshold)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}. "
                               "Supported formats: .csv, .xlsx, .xls")
                
        except Exception as e:
            raise RuntimeError(f"Failed to process file '{file_name}': {e}")
    

    
    def save_tables_to_csv(self, tables_list, output_dir="output", base_filename="extracted_table"):
        """Save extracted tables to CSV files."""
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        saved_files = []
        for i, (sheet_name, df) in enumerate(tables_list):
            clean_sheet_name = self._clean_filename(sheet_name)
            filename = f"{base_filename}.csv" if len(tables_list) == 1 else f"{base_filename}_{clean_sheet_name}.csv"
            filepath = os.path.join(output_dir, filename)
            
            try:
                df.to_csv(filepath, index=False, encoding='utf-8-sig')
                saved_files.append(filepath)
                print(f"Saved table '{sheet_name}' to: {filepath}")
            except Exception as e:
                print(f"Error saving table '{sheet_name}': {e}")
        
        return saved_files
    
    @staticmethod
    def _clean_filename(filename):
        """Clean filename by removing invalid characters."""
        import re
        cleaned = re.sub(r'[<>:"/\\|?*]', '_', str(filename))
        return cleaned.strip() if cleaned.strip() else "unnamed"

    @staticmethod
    def clean_table(df, max_null_threshold=0.8):
        """Deprecated: Simple table cleaning for backward compatibility with tests."""
        if df.empty:
            return df
        # Simple cleaning - just remove completely empty rows
        cleaned = df.dropna(how='all').reset_index(drop=True)
        return cleaned
        
    @staticmethod
    def filter_budget_tables(tables_list):
        """Deprecated: Simple budget filtering for backward compatibility with tests."""
        budget_tables = []
        for sheet_name, df in tables_list:
            if contains_money(df):
                budget_tables.append((sheet_name, df))
        return budget_tables
    
    def standardize_dataframe(self, df):
        """Standardize DataFrame for display."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Input is not a DataFrame. Type: {type(df)}")
        
        for col in df.columns:
            df[col] = df[col].astype('object').where(df[col].notna(), '')
        return df
    
    def preprocess_dataframe(self, df):
        """Clean up column names."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Input is not a DataFrame. Type: {type(df)}")
        
        df.columns = [f"Unnamed_{i}" if col == "" or col is None else str(col).strip() 
                      for i, col in enumerate(df.columns)]
        
        seen = set()
        new_columns = []
        for col in df.columns:
            original_col = col
            counter = 1
            while col in seen:
                col = f"{original_col}_{counter}"
                counter += 1
            seen.add(col)
            new_columns.append(col)
        
        df.columns = new_columns
        return df



def get_excel_processor():
    """Factory function to create an ExcelProcessor instance."""
    return ExcelProcessor()
