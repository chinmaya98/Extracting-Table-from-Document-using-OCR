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
            max_null_threshold: Float (0-1) indicating maximum proportion of null/empty cells allowed in a row
            
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
                    
                    if sheet_df.empty:
                        print(f"Warning: Sheet '{sheet_name}' is empty and will be skipped.")
                        metadata['empty_sheets'] += 1
                        continue
                    
                    # Fill NaN values with empty strings
                    sheet_df = sheet_df.fillna("")
                    
                    # Generate Excel-style column names (A, B, C, etc.)
                    excel_columns = self.generate_excel_column_names(len(sheet_df.columns))
                    sheet_df.columns = excel_columns
                    
                    # Set index to start from 1 (like Excel row numbers)
                    sheet_df.index = range(1, len(sheet_df) + 1)
                    
                    # Clean the sheet data (remove rows with too many null cells)
                    original_rows = len(sheet_df)
                    sheet_df = self.clean_table(sheet_df, max_null_threshold)
                    cleaned_rows = len(sheet_df)
                    
                    if original_rows > cleaned_rows:
                        print(f"Removed {original_rows - cleaned_rows} rows with high null content from sheet '{sheet_name}'")
                    
                    if not sheet_df.empty:
                        sheets[sheet_name] = sheet_df
                        tables.append((sheet_name, sheet_df))
                        metadata['processed_sheets'] += 1
                        
                        # Add sheet info to metadata
                        sheet_info = {
                            'sheet_name': sheet_name,
                            'rows': len(sheet_df),
                            'columns': len(sheet_df.columns),
                            'has_monetary_data': contains_money(sheet_df)
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
            max_null_threshold: Float (0-1) indicating maximum proportion of null/empty cells allowed in a row
            
        Returns:
            Tuple: (tables_list, sheets_dict, metadata)
        """
        try:
            # Read CSV file
            df = pd.read_csv(io.BytesIO(file_bytes))
            df = df.fillna("")  # Replace NaN with empty strings
            
            # Clean the data (remove rows with too many null cells)
            original_rows = len(df)
            df = self.clean_table(df, max_null_threshold)
            cleaned_rows = len(df)
            
            if original_rows > cleaned_rows:
                print(f"Removed {original_rows - cleaned_rows} rows with high null content from CSV file")
            
            # Create metadata
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
    
    @staticmethod
    def clean_table(df, max_null_threshold=0.8):
        """
        Remove empty/blank rows and rows with maximum null cells from a DataFrame.
        
        Args:
            df: pandas DataFrame to clean
            max_null_threshold: Float (0-1) indicating maximum proportion of null/empty cells allowed in a row
            
        Returns:
            Cleaned pandas DataFrame
        """
        if df.empty:
            return df
        
        # First, identify null/empty values (NaN, None, empty strings, whitespace-only strings)
        def is_null_or_empty(val):
            if pd.isna(val):
                return True
            if isinstance(val, str) and val.strip() == '':
                return True
            return False
        
        # Calculate the proportion of null/empty cells in each row
        null_counts = df.apply(lambda row: sum(is_null_or_empty(val) for val in row), axis=1)
        null_proportions = null_counts / len(df.columns)
        
        # Keep rows where null proportion is less than or equal to threshold
        df_cleaned = df[null_proportions <= max_null_threshold].copy()
        
        # Reset index
        df_cleaned = df_cleaned.reset_index(drop=True)
        
        # Fill remaining NaN values with empty strings
        df_cleaned = df_cleaned.fillna("")
        
        return df_cleaned
    
    @staticmethod
    def filter_budget_tables(tables_list):
        """
        Filter tables to only those containing monetary values.
        
        Args:
            tables_list: List of (sheet_name, DataFrame) tuples
            
        Returns:
            List of (sheet_name, DataFrame) tuples that contain monetary values
        """
        budget_tables = []
        for sheet_name, df in tables_list:
            if contains_money(df):
                budget_tables.append((sheet_name, df))
        return budget_tables
    
    @staticmethod
    def generate_excel_column_names(num_columns):
        """
        Generate Excel-style column names (A, B, C, ..., AA, AB, etc.)
        
        Args:
            num_columns: Number of columns to generate names for
            
        Returns:
            List of Excel-style column names
        """
        columns = []
        for i in range(num_columns):
            col_name = ""
            temp = i
            while temp >= 0:
                col_name = chr(65 + (temp % 26)) + col_name
                temp = temp // 26 - 1
                if temp < 0:
                    break
            columns.append(col_name)
        return columns
    
    def save_tables_to_csv(self, tables_list, output_dir="output", base_filename="extracted_table"):
        """
        Save extracted tables to CSV files.
        
        Args:
            tables_list: List of (sheet_name, DataFrame) tuples
            output_dir: Directory to save CSV files (default: "output")
            base_filename: Base filename for CSV files (default: "extracted_table")
            
        Returns:
            List of saved file paths
        """
        import os
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        saved_files = []
        
        for i, (sheet_name, df) in enumerate(tables_list):
            if df.empty:
                print(f"Skipping empty table '{sheet_name}'")
                continue
            
            # Clean sheet name for use as filename
            clean_sheet_name = self._clean_filename(sheet_name)
            
            # Generate filename
            if len(tables_list) == 1:
                filename = f"{base_filename}.csv"
            else:
                filename = f"{base_filename}_{clean_sheet_name}.csv"
            
            filepath = os.path.join(output_dir, filename)
            
            try:
                # Save to CSV
                df.to_csv(filepath, index=False, encoding='utf-8-sig')
                saved_files.append(filepath)
                print(f"Saved table '{sheet_name}' to: {filepath}")
                print(f"  - Rows: {len(df)}, Columns: {len(df.columns)}")
                
            except Exception as e:
                print(f"Error saving table '{sheet_name}' to CSV: {e}")
        
        return saved_files
    
    def save_all_tables_to_single_csv(self, tables_list, output_path="output/combined_tables.csv"):
        """
        Save all tables to a single CSV file with sheet names as a column.
        
        Args:
            tables_list: List of (sheet_name, DataFrame) tuples
            output_path: Path for the combined CSV file
            
        Returns:
            String: Path to the saved file or None if failed
        """
        import os
        
        if not tables_list:
            print("No tables to save")
            return None
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        combined_data = []
        
        for sheet_name, df in tables_list:
            if df.empty:
                continue
            
            # Add sheet name as a column
            df_copy = df.copy()
            df_copy.insert(0, 'Sheet_Name', sheet_name)
            combined_data.append(df_copy)
        
        if not combined_data:
            print("No non-empty tables to save")
            return None
        
        try:
            # Concatenate all dataframes
            combined_df = pd.concat(combined_data, ignore_index=True, sort=False)
            
            # Save to CSV
            combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"Saved combined tables to: {output_path}")
            print(f"  - Total rows: {len(combined_df)}, Columns: {len(combined_df.columns)}")
            
            return output_path
            
        except Exception as e:
            print(f"Error saving combined tables to CSV: {e}")
            return None
    
    @staticmethod
    def _clean_filename(filename):
        """
        Clean filename by removing or replacing invalid characters.
        
        Args:
            filename: Original filename string
            
        Returns:
            Cleaned filename string
        """
        import re
        # Remove or replace invalid characters for filename
        cleaned = re.sub(r'[<>:"/\\|?*]', '_', str(filename))
        cleaned = cleaned.strip()
        return cleaned if cleaned else "unnamed"
    
    def standardize_dataframe(self, df):
        """
        Ensure all columns in the DataFrame have consistent data types.
        
        Args:
            df: pandas DataFrame to standardize
            
        Returns:
            Standardized pandas DataFrame
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Input is not a valid DataFrame. Received type: {type(df)}")
        
        for col in df.columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
            
            # Use df.dtypes[col] to access the column's data type
            if df.dtypes[col] == 'object':
                # Convert mixed-type object columns to string
                df[col] = df[col].astype(str)
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Fill NaN values with 0 for numeric columns
                df[col] = df[col].fillna(0)
        
        return df
    
    def preprocess_dataframe(self, df):
        """
        Preprocess the DataFrame to clean up column names.
        
        Args:
            df: pandas DataFrame to preprocess
            
        Returns:
            Preprocessed pandas DataFrame
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Input is not a valid DataFrame. Received type: {type(df)}")
        
        # Replace empty or invalid column names
        df.columns = [
            f"Unnamed_{i}" if col == "" or col is None else str(col).strip()
            for i, col in enumerate(df.columns)
        ]
        
        # Ensure all column names are unique
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
    
    def process_and_save_to_csv(self, file_bytes, file_extension, file_name="", 
                               output_dir="output", max_null_threshold=0.8, 
                               save_combined=False):
        """
        Process Excel/CSV file and automatically save results to CSV files.
        
        Args:
            file_bytes: File content as bytes
            file_extension: File extension (e.g., '.xlsx', '.csv')
            file_name: Original filename
            output_dir: Directory to save CSV files
            max_null_threshold: Float (0-1) indicating maximum proportion of null/empty cells allowed in a row
            save_combined: Whether to also save a combined CSV file
            
        Returns:
            Dictionary with processing results and saved file paths
        """
        try:
            # Process the file
            tables, sheets, metadata = self.process_file(
                file_bytes, file_extension, file_name, max_null_threshold
            )
            
            if not tables:
                return {
                    'success': False,
                    'message': 'No valid tables found in the file',
                    'metadata': metadata,
                    'saved_files': []
                }
            
            # Generate base filename from original filename
            import os
            base_name = os.path.splitext(file_name)[0] if file_name else "extracted_table"
            base_name = self._clean_filename(base_name)
            
            # Save individual CSV files
            saved_files = self.save_tables_to_csv(tables, output_dir, base_name)
            
            # Save combined CSV file if requested
            combined_file = None
            if save_combined and len(tables) > 1:
                combined_path = os.path.join(output_dir, f"{base_name}_combined.csv")
                combined_file = self.save_all_tables_to_single_csv(tables, combined_path)
                if combined_file:
                    saved_files.append(combined_file)
            
            return {
                'success': True,
                'message': f'Successfully processed {len(tables)} table(s)',
                'metadata': metadata,
                'saved_files': saved_files,
                'tables_count': len(tables),
                'combined_file': combined_file
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error processing file: {str(e)}',
                'metadata': {},
                'saved_files': []
            }
    
    def get_sheets_summary(self, sheets_dict):
        """
        Get summary information about all sheets.
        
        Args:
            sheets_dict: Dictionary of {sheet_name: DataFrame}
            
        Returns:
            Dictionary with sheets summary
        """
        summary = {
            'total_sheets': len(sheets_dict),
            'sheets': []
        }
        
        for sheet_name, df in sheets_dict.items():
            sheet_summary = {
                'name': sheet_name,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'has_monetary_data': contains_money(df),
                'is_empty': df.empty
            }
            summary['sheets'].append(sheet_summary)
        
        return summary


def get_excel_processor():
    """
    Factory function to create an ExcelProcessor instance.
    
    Returns:
        ExcelProcessor instance
    """
    return ExcelProcessor()


# Example usage
if __name__ == "__main__":
    # Example of how to use the Excel processor with CSV output
    try:
        processor = get_excel_processor()
        
        # Example with an Excel file
        with open("sample.xlsx", "rb") as f:
            excel_bytes = f.read()
        
        # Process file with null cell filtering and save to CSV
        result = processor.process_and_save_to_csv(
            excel_bytes, 
            ".xlsx", 
            "sample.xlsx",
            output_dir="output",
            max_null_threshold=0.7,  # Remove rows with more than 70% null cells
            save_combined=True
        )
        
        if result['success']:
            print(f"✅ {result['message']}")
            print(f"📊 Processed {result['metadata']['processed_sheets']} sheets")
            print("📁 Saved files:")
            for file_path in result['saved_files']:
                print(f"   - {file_path}")
        else:
            print(f"❌ {result['message']}")
        
        # Alternative: Manual process and save
        # tables, sheets, metadata = processor.process_file(
        #     excel_bytes, ".xlsx", "sample.xlsx", max_null_threshold=0.8
        # )
        # 
        # print(f"Processed {metadata['processed_sheets']} sheets from Excel file")
        # for sheet_info in metadata['sheet_info']:
        #     print(f"Sheet '{sheet_info['sheet_name']}': {sheet_info['rows']} rows, {sheet_info['columns']} columns")
        # 
        # # Save to CSV files
        # saved_files = processor.save_tables_to_csv(tables, "output", "my_extracted_data")
        # print(f"Saved {len(saved_files)} CSV files")
        # 
        # # Get budget-related tables
        # budget_tables = processor.filter_budget_tables(tables)
        # print(f"Found {len(budget_tables)} sheets with monetary data")
        
    except FileNotFoundError:
        print("❌ Sample file not found. Please provide a valid Excel file.")
    except Exception as e:
        print(f"❌ Error: {e}")
        
    # Example with CSV file
    try:
        print("\n" + "="*50)
        print("Processing CSV file example:")
        
        with open("sample.csv", "rb") as f:
            csv_bytes = f.read()
        
        result = processor.process_and_save_to_csv(
            csv_bytes,
            ".csv",
            "sample.csv",
            max_null_threshold=0.8
        )
        
        if result['success']:
            print(f"✅ {result['message']}")
            print("📁 Saved files:")
            for file_path in result['saved_files']:
                print(f"   - {file_path}")
        else:
            print(f"❌ {result['message']}")
            
    except FileNotFoundError:
        print("ℹ️  CSV sample file not found - skipping CSV example.")
    except Exception as e:
        print(f"❌ CSV Error: {e}")
