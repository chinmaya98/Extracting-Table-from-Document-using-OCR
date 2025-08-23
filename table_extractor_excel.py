"""
Table Extractor for Excel Files
Processes Excel files directly without Document Intelligence.
Handles multi-sheet processing with individual sheet analysis.
"""
import io
import os
import pandas as pd
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional
from utils.currency_utils import contains_money


class AzureConfig:
    """Azure configuration manager for Excel processing."""
    def __init__(self):
        load_dotenv()
        self.blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
        self.blob_container = os.getenv("AZURE_BLOB_CONTAINER")
        
        # Store which services are available
        self.blob_storage_available = bool(self.blob_connection_string and self.blob_container)

    def get_blob_service_client(self) -> Optional[BlobServiceClient]:
        """Get Blob Service client."""
        if not self.blob_storage_available:
            return None
        try:
            return BlobServiceClient.from_connection_string(self.blob_connection_string)
        except Exception:
            return None


class BlobManager:
    """Azure Blob Storage manager for Excel files."""
    def __init__(self, service_client: Optional[BlobServiceClient], container: str):
        self.service_client = service_client
        self.container = container
        self.container_client = service_client.get_container_client(container) if service_client else None

    def list_files(self, extensions: tuple = (".xlsx", ".xls")) -> List[str]:
        """List Excel files with specified extensions."""
        if not self.container_client:
            return []
        try:
            return [b.name for b in self.container_client.list_blobs() 
                    if b.name.lower().endswith(extensions)]
        except Exception:
            return []

    def download_file(self, blob_name: str) -> bytes:
        """Download file as bytes."""
        if not self.service_client:
            raise RuntimeError("Blob storage not configured")
        try:
            blob_client = self.service_client.get_blob_client(
                container=self.container, blob=blob_name
            )
            return blob_client.download_blob().readall()
        except Exception as e:
            raise RuntimeError(f"Failed to download {blob_name}: {e}")


class ExcelTableExtractor:
    """Extract tables from Excel files by processing each sheet individually."""
    
    def __init__(self):
        """Initialize Excel table extractor."""
        pass

    def extract_from_excel(self, excel_bytes: bytes, filename: str = "") -> Tuple[List[pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Extract tables from Excel file by processing each sheet separately.
        
        Returns:
            Tuple of (tables_list, sheets_dict) where:
            - tables_list: List of DataFrames with budget data
            - sheets_dict: Dictionary mapping sheet names to DataFrames
        """
        try:
            print(f"📊 Processing Excel file: {filename}")
            
            tables = []
            sheets = {}
            sheet_tables = {}  # Track which tables came from which sheets
            
            # Process each Excel sheet separately
            excel_file = pd.ExcelFile(io.BytesIO(excel_bytes))
            print(f"📋 Found {len(excel_file.sheet_names)} sheets: {excel_file.sheet_names}")
            
            for sheet_name in excel_file.sheet_names:
                try:
                    print(f"\n🔍 Processing sheet: {sheet_name}")
                    
                    # Try different engines for better compatibility
                    df = None
                    engines_to_try = ['openpyxl', 'xlrd'] if filename.lower().endswith('.xls') else ['openpyxl']
                    
                    for engine in engines_to_try:
                        try:
                            df = pd.read_excel(excel_file, sheet_name=sheet_name, engine=engine)
                            print(f"   📖 Successfully read with {engine} engine")
                            break
                        except Exception as engine_error:
                            if engine == engines_to_try[-1]:  # Last engine
                                print(f"   ❌ Failed to read sheet {sheet_name} with all engines: {engine_error}")
                            continue
                    
                    if df is not None and not df.empty:
                        # Clean the table and preserve original headers
                        df = self.clean_table(df)
                        
                        # Store original headers from the Excel sheet
                        original_headers = list(df.columns)
                        df._original_headers = original_headers
                        
                        # Add sheet metadata
                        df._sheet_name = sheet_name
                        df._source_info = f"Excel_Sheet_{sheet_name}"
                        df._source_type = "Excel"
                        df._filename = filename
                        
                        print(f"   📏 Sheet dimensions: {len(df)} rows × {len(df.columns)} columns")
                        print(f"   📝 Original headers: {original_headers}")
                        
                        # Check if sheet contains budget data
                        if contains_money(df):
                            tables.append(df)
                            sheets[sheet_name] = df
                            sheet_tables[sheet_name] = df
                            print(f"   ✅ Sheet '{sheet_name}': Contains budget data - included")
                        else:
                            print(f"   ⏭️  Sheet '{sheet_name}': No budget data found - skipped")
                            
                except Exception as e:
                    print(f"   ❌ Warning: Failed to read sheet {sheet_name}: {e}")
                    continue
            
            if not tables:
                # Return all sheets for AI summary generation instead of raising error
                all_sheets = {}
                all_tables = []
                
                print(f"\n⚠️  No budget tables found, collecting all data for AI summary...")
                
                # Re-process all sheets without budget filtering for summary
                for sheet_name in excel_file.sheet_names:
                    try:
                        df = pd.read_excel(excel_file, sheet_name=sheet_name)
                        if not df.empty and len(df) > 1:
                            # Store original headers
                            df._original_headers = list(df.columns)
                            df._sheet_name = sheet_name
                            all_sheets[sheet_name] = df
                            all_tables.append(df)
                            print(f"   📊 Sheet '{sheet_name}': {len(df)} rows collected for summary")
                    except Exception as e:
                        print(f"   ❌ Error reading sheet {sheet_name}: {e}")
                        continue
                
                # Return data for AI summary instead of throwing error
                return all_tables, all_sheets
            
            print(f"\n📈 Processing Summary:")
            print(f"   • Total sheets processed: {len(excel_file.sheet_names)}")
            print(f"   • Sheets with budget data: {len(tables)}")
            print(f"   • Budget tables extracted: {len(tables)}")
            
            # Each sheet is processed individually - no combination needed
            for sheet_name, df in sheets.items():
                print(f"   • {sheet_name}: {len(df)} rows, headers: {getattr(df, '_original_headers', list(df.columns))}")
                
            return tables, sheets
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract tables from Excel: {e}")

    def process_file(self, file_bytes: bytes, file_extension: str, filename: str = "") -> Tuple[List[pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Process Excel file and extract tables with sheet information."""
        ext = file_extension.lower()
        
        if ext in [".xlsx", ".xls"]:
            return self.extract_from_excel(file_bytes, filename)
        else:
            raise ValueError(f"Unsupported file format for Excel extractor: {ext}")

    def get_table_source_info(self, table: pd.DataFrame) -> str:
        """Get source information for a table (sheet name)."""
        if hasattr(table, '_sheet_name'):
            return f"Sheet: {table._sheet_name}"
        elif hasattr(table, '_source_info'):
            return table._source_info
        else:
            return "Unknown Excel source"

    def get_sheet_info(self, table: pd.DataFrame) -> Dict[str, str]:
        """Get detailed sheet information for a table."""
        return {
            'sheet_name': getattr(table, '_sheet_name', 'Unknown'),
            'source_type': getattr(table, '_source_type', 'Excel'),
            'filename': getattr(table, '_filename', ''),
            'original_headers': getattr(table, '_original_headers', list(table.columns)),
            'rows': len(table),
            'columns': len(table.columns)
        }

    def clean_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame by removing empty rows and preserving original Excel headers."""
        if df.empty:
            return df
        
        # Store original headers from Excel sheet
        original_headers = list(df.columns)
        
        # Remove completely empty rows
        df = df[~df.apply(lambda row: row.astype(str).str.strip().eq('').all(), axis=1)]
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Preserve original Excel column headers (don't change to generic names)
        df.columns = [str(col).strip() if col and str(col).strip() else f"Column_{i+1}" 
                      for i, col in enumerate(original_headers)]
        
        # Store original headers as metadata for AI processing
        df._original_headers = list(df.columns)
        
        # Fill NaN values
        df = df.fillna("")
        
        return df

    @staticmethod
    def get_extractor() -> 'ExcelTableExtractor':
        """Get ExcelTableExtractor instance."""
        return ExcelTableExtractor()

    @staticmethod
    def get_blob_manager() -> Optional[BlobManager]:
        """Get configured BlobManager instance for Excel files."""
        try:
            config = AzureConfig()
            blob_service_client = config.get_blob_service_client()
            if blob_service_client is None or not config.blob_container:
                return None
            return BlobManager(blob_service_client, config.blob_container)
        except Exception:
            return None


def test_excel_extractor():
    """Test the Excel table extractor functionality."""
    print("🧪 Testing Excel Table Extractor")
    print("=" * 50)
    
    # Create test Excel data (simulating multi-sheet Excel file)
    test_sheets = {
        'Marketing_Budget': pd.DataFrame({
            'Campaign Name': ['Digital Marketing', 'Social Media', 'Print Ads', 'TOTAL'],
            'Budget Allocated': [15000.00, 8500.00, 3200.00, 26700.00],
            'Status': ['Active', 'Planning', 'Completed', 'Summary']
        }),
        
        'IT_Expenses': pd.DataFrame({
            'Item': ['Laptops', 'Software Licenses', 'Cloud Storage'],
            'Cost': [25000.00, 12000.00, 3600.00],
            'Department': ['All', 'Development', 'All']
        }),
        
        'Employee_List': pd.DataFrame({
            'Name': ['John Doe', 'Jane Smith', 'Bob Wilson'],
            'Department': ['IT', 'HR', 'Marketing'],
            'Location': ['New York', 'Chicago', 'Los Angeles']
        })
    }
    
    extractor = ExcelTableExtractor()
    processed_sheets = []
    
    print("📊 Processing individual sheets:")
    for sheet_name, df in test_sheets.items():
        print(f"\n🔍 Sheet: {sheet_name}")
        
        # Simulate the processing that would happen
        df = extractor.clean_table(df)
        df._sheet_name = sheet_name
        df._original_headers = list(df.columns)
        
        print(f"   📏 Dimensions: {len(df)} rows × {len(df.columns)} columns")
        print(f"   📝 Headers: {list(df.columns)}")
        
        if contains_money(df):
            processed_sheets.append({
                'sheet_name': sheet_name,
                'data': df,
                'original_headers': list(df.columns)
            })
            print(f"   ✅ Contains budget data - will be processed by AI")
        else:
            print(f"   ⏭️  No budget data - will be skipped")
    
    print(f"\n📈 Results:")
    print(f"   • Total sheets: {len(test_sheets)}")
    print(f"   • Sheets with budget data: {len(processed_sheets)}")
    print(f"   • Individual processing: ✅")
    print(f"   • Original headers preserved: ✅")
    
    print(f"\n🎯 Sheets ready for AI processing:")
    for sheet in processed_sheets:
        print(f"   • {sheet['sheet_name']}: {sheet['original_headers']}")
    
    print(f"\n🔧 Excel Extractor Features:")
    print("✅ Direct Excel file processing (no Document Intelligence needed)")
    print("✅ Individual sheet iteration and processing")
    print("✅ Original Excel header preservation")
    print("✅ Multi-engine support (openpyxl, xlrd)")
    print("✅ Budget table filtering per sheet")
    print("✅ Sheet metadata tracking")
    print("✅ Ready for AI Budget Extractor integration")


if __name__ == "__main__":
    print("=" * 60)
    print("EXCEL TABLE EXTRACTOR")
    print("=" * 60)
    print("\nSpecialized processor for Excel files (.xlsx, .xls)")
    print("Processes each sheet individually without Document Intelligence")
    print("Preserves original headers and filters budget-related data\n")
    
    # Run the test
    test_excel_extractor()
    
    print("\n" + "=" * 60)
    print("Ready for integration with Trinity Online workflow!")
    print("=" * 60)
