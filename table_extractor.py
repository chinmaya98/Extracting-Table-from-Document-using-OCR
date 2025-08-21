"""
Table Extractor using Azure Document Intelligence Layout Model
Extracts tables from PDFs, images, and Excel files.
"""
import io
import os
import pandas as pd
from PIL import Image
import filetype
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
        """Extract tables from Excel file."""
        try:
            excel_file = pd.ExcelFile(io.BytesIO(excel_bytes))
            tables = []
            sheets = {}
            
            for sheet_name in excel_file.sheet_names:
                try:
                    # Read with headers
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    if not df.empty:
                        df = self.clean_table(df)
                        if contains_money(df):
                            tables.append(df)
                            sheets[sheet_name] = df
                except Exception as e:
                    print(f"Warning: Failed to read sheet {sheet_name}: {e}")
                    continue
                    
            return tables, sheets
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract tables from Excel: {e}")

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
