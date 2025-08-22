"""
Table Extractor using Azure Document Intelligence Layout Model
Extracts tables from PDFs, images, and Excel files.
"""
import io
import os
import pandas as pd
from PIL import Image
import filetype
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from typing import List, Tuple, Optional
from utils.currency_utils import contains_money


class AzureConfig:
    """Azure configuration manager."""
    def __init__(self):
        load_dotenv()
        self.endpoint = os.getenv("DOC_INTELLIGENCE_ENDPOINT")
        self.key = os.getenv("DOC_INTELLIGENCE_KEY")
        self.blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
        self.blob_container = os.getenv("AZURE_BLOB_CONTAINER")
        
        if not all([self.endpoint, self.key, self.blob_connection_string, self.blob_container]):
            raise ValueError("Missing required Azure environment variables")

    def get_document_client(self) -> DocumentIntelligenceClient:
        """Get Document Intelligence client."""
        return DocumentIntelligenceClient(
            endpoint=self.endpoint, 
            credential=AzureKeyCredential(self.key)
        )

    def get_blob_service_client(self) -> BlobServiceClient:
        """Get Blob Service client."""
        return BlobServiceClient.from_connection_string(self.blob_connection_string)


class BlobManager:
    """Azure Blob Storage manager."""
    def __init__(self, service_client: BlobServiceClient, container: str):
        self.service_client = service_client
        self.container = container
        self.container_client = service_client.get_container_client(container)

    def list_files(self, extensions: tuple = (".pdf", ".xlsx", ".xls", ".jpg", ".jpeg", ".png", ".tiff", ".bmp")) -> List[str]:
        """List files with specified extensions."""
        return [b.name for b in self.container_client.list_blobs() 
                if b.name.lower().endswith(extensions)]

    def download_file(self, blob_name: str) -> bytes:
        """Download file as bytes."""
        blob_client = self.service_client.get_blob_client(
            container=self.container, blob=blob_name
        )
        return blob_client.download_blob().readall()


class TableExtractor:
    """Extract tables from various file formats using Azure Document Intelligence."""
    
    def __init__(self, document_client: DocumentIntelligenceClient):
        self.client = document_client

    def extract_from_pdf(self, pdf_bytes: bytes) -> List[pd.DataFrame]:
        """Extract tables from PDF using Document Intelligence layout model."""
        try:
            poller = self.client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=pdf_bytes,
                content_type="application/pdf"
            )
            result = poller.result()
            
            tables = []
            for table in result.tables:
                df = self._process_table_structure(table)
                if not df.empty and contains_money(df):
                    tables.append(df)
            return tables
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract tables from PDF: {e}")

    def extract_from_image(self, image_bytes: bytes) -> List[pd.DataFrame]:
        """Extract tables from image using Document Intelligence layout model."""
        try:
            # Detect image type
            kind = filetype.guess(image_bytes)
            if kind is None or not kind.mime.startswith("image/"):
                raise ValueError("Unsupported or undetectable image type")
            
            poller = self.client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=image_bytes,
                content_type=kind.mime
            )
            result = poller.result()
            
            tables = []
            for table in result.tables:
                df = self._process_table_structure(table)
                if not df.empty and contains_money(df):
                    tables.append(df)
            return tables
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract tables from image: {e}")

    def extract_from_excel(self, excel_bytes: bytes, file_name: str = "") -> Tuple[List[pd.DataFrame], dict]:
        """Extract tables from Excel file by converting to PDF first."""
        try:
            # First, try to convert Excel to PDF and extract via PDF pipeline
            try:
                pdf_bytes = self._convert_excel_to_pdf(excel_bytes)
                pdf_tables = self.extract_from_pdf(pdf_bytes)
            except Exception as pdf_error:
                print(f"Warning: PDF conversion failed ({pdf_error}), falling back to direct Excel processing")
                pdf_tables = []
            
            # Always get original Excel data as reference/fallback
            tables = []
            sheets = {}
            
            try:
                excel_file = pd.ExcelFile(io.BytesIO(excel_bytes))
                
                for sheet_name in excel_file.sheet_names:
                    try:
                        # Try different engines for better compatibility
                        df = None
                        engines_to_try = ['openpyxl', 'xlrd'] if file_name.lower().endswith('.xls') else ['openpyxl']
                        
                        for engine in engines_to_try:
                            try:
                                df = pd.read_excel(excel_file, sheet_name=sheet_name, engine=engine)
                                break
                            except Exception as engine_error:
                                if engine == engines_to_try[-1]:  # Last engine
                                    print(f"Warning: Failed to read sheet {sheet_name} with all engines: {engine_error}")
                                continue
                        
                        if df is not None and not df.empty:
                            df = self.clean_table(df)
                            if contains_money(df):
                                tables.append(df)
                                sheets[sheet_name] = df
                                
                    except Exception as e:
                        print(f"Warning: Failed to read sheet {sheet_name}: {e}")
                        continue
                        
            except Exception as excel_error:
                print(f"Warning: Excel processing failed: {excel_error}")
            
            # Combine PDF extracted tables (if any) with direct Excel tables
            final_tables = pdf_tables if pdf_tables else tables
            
            if not final_tables and not sheets:
                raise RuntimeError("No tables could be extracted from Excel file")
                
            return final_tables, sheets
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract tables from Excel: {e}")

    def _convert_excel_to_pdf(self, excel_bytes: bytes) -> bytes:
        """Convert Excel file to PDF for consistent table extraction."""
        try:
            # Try different methods to read Excel file
            excel_file = None
            
            try:
                excel_file = pd.ExcelFile(io.BytesIO(excel_bytes))
            except Exception as e:
                # If xlrd is missing for .xls files, try with openpyxl
                if "xlrd" in str(e).lower():
                    try:
                        # Try converting bytes to see if it's actually .xlsx
                        excel_file = pd.ExcelFile(io.BytesIO(excel_bytes), engine='openpyxl')
                    except Exception:
                        raise RuntimeError(f"Cannot read Excel file. For .xls files, xlrd>=2.0.1 is required. For .xlsx files, openpyxl is required. Error: {e}")
                else:
                    raise e
            
            if not excel_file:
                raise RuntimeError("Could not read Excel file with any available engine")
            
            # Create PDF in memory
            pdf_buffer = io.BytesIO()
            
            with PdfPages(pdf_buffer, keep_empty=False) as pdf:
                for sheet_name in excel_file.sheet_names:
                    try:
                        # Try reading with appropriate engine
                        df = None
                        
                        # Determine engines to try based on file type hints
                        engines_to_try = ['openpyxl', 'xlrd']
                        
                        for engine in engines_to_try:
                            try:
                                df = pd.read_excel(excel_file, sheet_name=sheet_name, engine=engine)
                                break
                            except Exception as engine_error:
                                if "xlrd" in str(engine_error).lower() and engine == 'xlrd':
                                    # xlrd not available, skip this engine
                                    continue
                                elif engine == engines_to_try[-1]:  # Last engine failed
                                    print(f"Warning: Could not read sheet {sheet_name} with any engine: {engine_error}")
                                    break
                        
                        if df is None or df.empty:
                            continue
                            
                        # Create a new figure for each sheet
                        fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 landscape
                        ax.axis('tight')
                        ax.axis('off')
                        
                        # Add sheet title
                        fig.suptitle(f'Sheet: {sheet_name}', fontsize=16, fontweight='bold')
                        
                        # Convert DataFrame to table format that looks like Excel
                        # Handle large DataFrames by chunking if necessary
                        max_rows_per_page = 25
                        max_cols_per_page = 8
                        
                        # Split large tables into chunks
                        for row_start in range(0, len(df), max_rows_per_page):
                            for col_start in range(0, len(df.columns), max_cols_per_page):
                                row_end = min(row_start + max_rows_per_page, len(df))
                                col_end = min(col_start + max_cols_per_page, len(df.columns))
                                
                                df_chunk = df.iloc[row_start:row_end, col_start:col_end]
                                
                                if df_chunk.empty:
                                    continue
                                
                                # Create new page for each chunk if not first chunk
                                if row_start > 0 or col_start > 0:
                                    fig, ax = plt.subplots(figsize=(11.69, 8.27))
                                    ax.axis('tight')
                                    ax.axis('off')
                                    fig.suptitle(f'Sheet: {sheet_name} (Part {row_start//max_rows_per_page + 1}-{col_start//max_cols_per_page + 1})', 
                                               fontsize=14)
                                
                                # Convert DataFrame values to strings and handle NaN
                                table_data = df_chunk.fillna('').astype(str)
                                
                                # Create table
                                table = ax.table(
                                    cellText=table_data.values,
                                    colLabels=table_data.columns,
                                    cellLoc='center',
                                    loc='center',
                                    bbox=[0.1, 0.1, 0.8, 0.8]
                                )
                                
                                # Style the table to look professional
                                table.auto_set_font_size(False)
                                table.set_fontsize(9)
                                table.scale(1, 1.5)
                                
                                # Style header row
                                for i in range(len(table_data.columns)):
                                    table[(0, i)].set_facecolor('#4CAF50')
                                    table[(0, i)].set_text_props(weight='bold', color='white')
                                
                                # Style data cells
                                for i in range(1, len(table_data) + 1):
                                    for j in range(len(table_data.columns)):
                                        if i % 2 == 0:
                                            table[(i, j)].set_facecolor('#f0f0f0')
                                        else:
                                            table[(i, j)].set_facecolor('white')
                                
                                # Save this page
                                pdf.savefig(fig, bbox_inches='tight', dpi=150)
                                plt.close(fig)
                        
                    except Exception as sheet_error:
                        print(f"Warning: Failed to convert sheet {sheet_name} to PDF: {sheet_error}")
                        continue
            
            pdf_buffer.seek(0)
            pdf_bytes = pdf_buffer.read()
            pdf_buffer.close()
            
            if len(pdf_bytes) == 0:
                raise ValueError("Failed to generate PDF from Excel file - no readable sheets found")
                
            return pdf_bytes
            
        except Exception as e:
            raise RuntimeError(f"Failed to convert Excel to PDF: {e}")

    def process_file(self, file_bytes: bytes, file_extension: str) -> List[pd.DataFrame]:
        """Process file based on extension and extract tables."""
        ext = file_extension.lower()
        
        if ext == ".pdf":
            return self.extract_from_pdf(file_bytes)
        elif ext in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
            return self.extract_from_image(file_bytes)
        elif ext in [".xlsx", ".xls"]:
            tables, _ = self.extract_from_excel(file_bytes)
            return tables
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _process_table_structure(self, table) -> pd.DataFrame:
        """Process Azure Document Intelligence table structure into DataFrame."""
        try:
            if not hasattr(table, 'row_count') or not hasattr(table, 'column_count'):
                return pd.DataFrame()
                
            rows = table.row_count
            cols = table.column_count
            
            if rows == 0 or cols == 0:
                return pd.DataFrame()
            
            # Initialize grid
            grid = [["" for _ in range(cols)] for _ in range(rows)]
            
            # Fill grid with cell contents
            for cell in table.cells:
                if (0 <= cell.row_index < rows and 0 <= cell.column_index < cols):
                    grid[cell.row_index][cell.column_index] = str(cell.content or "").strip()
            
            # Create DataFrame - use first row as headers if it exists
            if rows > 1:
                headers = grid[0]
                data = grid[1:]
                df = pd.DataFrame(data, columns=headers)
            else:
                df = pd.DataFrame(grid)
            
            return self.clean_table(df)
            
        except Exception as e:
            print(f"Warning: Failed to process table structure: {e}")
            return pd.DataFrame()

    def clean_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame by removing empty rows and standardizing data."""
        if df.empty:
            return df
            
        # Remove completely empty rows
        df = df[~df.apply(lambda row: row.astype(str).str.strip().eq('').all(), axis=1)]
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Clean column names
        df.columns = [str(col).strip() if col else f"Column_{i}" 
                      for i, col in enumerate(df.columns)]
        
        # Fill NaN values
        df = df.fillna("")
        
        return df

    @staticmethod
    def get_extractor() -> 'TableExtractor':
        """Get configured TableExtractor instance."""
        config = AzureConfig()
        document_client = config.get_document_client()
        return TableExtractor(document_client)

    @staticmethod
    def get_blob_manager() -> BlobManager:
        """Get configured BlobManager instance."""
        config = AzureConfig()
        blob_service_client = config.get_blob_service_client()
        return BlobManager(blob_service_client, config.blob_container)


def test_table_filtering():
    """Test that tables are properly filtered based on money content."""
    
    # Create a sample table extractor instance (without needing Azure config for this test)
    class MockTableExtractor:
        def clean_table(self, df):
            # Simple cleaning for test
            if df.empty:
                return df
            # Remove completely empty rows
            df = df[~df.apply(lambda row: row.astype(str).str.strip().eq('').all(), axis=1)]
            # Reset index
            df = df.reset_index(drop=True)
            # Clean column names
            df.columns = [str(col).strip() if col else f"Column_{i}" 
                          for i, col in enumerate(df.columns)]
            # Fill NaN values
            df = df.fillna("")
            return df
        
        def test_filtering_logic(self, tables_data):
            """Test the filtering logic that would be applied in the actual methods."""
            filtered_tables = []
            
            for table_data in tables_data:
                df = pd.DataFrame(table_data)
                df = self.clean_table(df)
                if not df.empty and contains_money(df):
                    filtered_tables.append(df)
                    
            return filtered_tables
    
    # Test data - mix of tables with and without money values
    test_tables = [
        # Table 1: Contains money values - SHOULD BE INCLUDED
        {
            'Item': ['Laptop', 'Mouse', 'Keyboard'],
            'Price': ['$1,200.00', '$25.99', '$89.50'],
            'Quantity': [1, 2, 1]
        },
        
        # Table 2: No money values - SHOULD BE FILTERED OUT
        {
            'Employee': ['John Doe', 'Jane Smith', 'Bob Wilson'],
            'Department': ['IT', 'HR', 'Marketing'],
            'Location': ['New York', 'Chicago', 'Los Angeles']
        },
        
        # Table 3: Contains budget/amount keywords - SHOULD BE INCLUDED
        {
            'Category': ['Travel', 'Equipment', 'Training'],
            'Budget Amount': ['5000', '15000', '3000'],
            'Status': ['Approved', 'Pending', 'Approved']
        },
        
        # Table 4: Contains currency symbols - SHOULD BE INCLUDED  
        {
            'Product': ['Widget A', 'Widget B'],
            'Cost': ['₹2,500', '€150'],
            'Profit': ['₹500', '€30']
        },
        
        # Table 5: Just text data - SHOULD BE FILTERED OUT
        {
            'Name': ['Project Alpha', 'Project Beta'],
            'Status': ['In Progress', 'Completed'],
            'Team': ['Team A', 'Team B']
        }
    ]
    
    # Test the filtering
    extractor = MockTableExtractor()
    filtered_results = extractor.test_filtering_logic(test_tables)
    
    print(f"Original tables: {len(test_tables)}")
    print(f"Filtered tables (with money): {len(filtered_results)}")
    print(f"Expected filtered tables: 3")
    
    # Verify results
    assert len(filtered_results) == 3, f"Expected 3 tables with money, got {len(filtered_results)}"
    
    # Print details of filtered tables
    for i, df in enumerate(filtered_results, 1):
        print(f"\nTable {i} (kept):")
        print(df.to_string())
    
    print("\n✅ Test passed! Tables without amount values are properly filtered out.")
    print("The filtering logic will now:")
    print("- Keep tables with currency symbols ($, €, ₹, etc.)")
    print("- Keep tables with amount/budget keywords in headers")
    print("- Keep tables with numeric currency patterns")
    print("- Filter out tables with only text data")


if __name__ == "__main__":
    print("=" * 60)
    print("TABLE EXTRACTOR - Money-Based Filtering System")
    print("=" * 60)
    print("\nThis module extracts tables from PDFs, images, and Excel files")
    print("and automatically filters out tables that don't contain monetary values.\n")
    
    # Run the test
    test_table_filtering()
    
    print("\n" + "=" * 60)
    print("Ready to use! Import TableExtractor for table extraction.")
    print("=" * 60)
