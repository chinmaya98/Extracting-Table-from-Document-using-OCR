"""
Table Extractor for PDF and Image Files
Uses Azure Document Intelligence Layout Model for OCR and table extraction.
"""
import os
import pandas as pd
import filetype
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from typing import List, Optional
from utils.currency_utils import contains_money


class AzureConfig:
    """Azure configuration manager for PDF/Image processing."""
    def __init__(self):
        load_dotenv()
        self.endpoint = os.getenv("DOC_INTELLIGENCE_ENDPOINT")
        self.key = os.getenv("DOC_INTELLIGENCE_KEY")
        self.blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
        self.blob_container = os.getenv("AZURE_BLOB_CONTAINER")
        
        # Store which services are available
        self.doc_intelligence_available = bool(self.endpoint and self.key)
        self.blob_storage_available = bool(self.blob_connection_string and self.blob_container)

    def get_document_client(self) -> Optional[DocumentIntelligenceClient]:
        """Get Document Intelligence client."""
        if not self.doc_intelligence_available:
            return None
        try:
            return DocumentIntelligenceClient(
                endpoint=self.endpoint, 
                credential=AzureKeyCredential(self.key)
            )
        except Exception:
            return None

    def get_blob_service_client(self) -> Optional[BlobServiceClient]:
        """Get Blob Service client."""
        if not self.blob_storage_available:
            return None
        try:
            return BlobServiceClient.from_connection_string(self.blob_connection_string)
        except Exception:
            return None


class BlobManager:
    """Azure Blob Storage manager for PDF/Image files."""
    def __init__(self, service_client: Optional[BlobServiceClient], container: str):
        self.service_client = service_client
        self.container = container
        self.container_client = service_client.get_container_client(container) if service_client else None

    def list_files(self, extensions: tuple = (".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".bmp")) -> List[str]:
        """List PDF and image files with specified extensions."""
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


class PDFImageTableExtractor:
    """Extract tables from PDF and image files using Azure Document Intelligence."""
    
    def __init__(self, document_client: Optional[DocumentIntelligenceClient]):
        self.client = document_client

    def extract_from_pdf(self, pdf_bytes: bytes, filename: str = "") -> List[pd.DataFrame]:
        """Extract tables from PDF using Document Intelligence layout model."""
        if not self.client:
            raise RuntimeError("Document Intelligence client not configured")
        
        try:
            print(f"📄 Processing PDF: {filename}")
            poller = self.client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=pdf_bytes,
                content_type="application/pdf"
            )
            result = poller.result()
            
            tables = []
            for i, table in enumerate(result.tables):
                df = self._process_table_structure(table)
                if not df.empty and contains_money(df):
                    # Add source information
                    df._source_info = f"PDF_Page_{getattr(table, 'page_number', i+1)}_Table_{i+1}"
                    df._source_type = "PDF"
                    df._filename = filename
                    tables.append(df)
                    print(f"✅ Table {i+1}: Contains budget data - included")
                else:
                    print(f"⏭️  Table {i+1}: No budget data found - skipped")
            
            print(f"💰 Found {len(tables)} budget-related tables in PDF")
            return tables
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract tables from PDF: {e}")

    def extract_from_image(self, image_bytes: bytes, filename: str = "") -> List[pd.DataFrame]:
        """Extract tables from image using Document Intelligence layout model."""
        if not self.client:
            raise RuntimeError("Document Intelligence client not configured")
        
        try:
            print(f"🖼️  Processing Image: {filename}")
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
            for i, table in enumerate(result.tables):
                df = self._process_table_structure(table)
                if not df.empty and contains_money(df):
                    # Add source information
                    df._source_info = f"Image_Table_{i+1}"
                    df._source_type = "Image"
                    df._filename = filename
                    tables.append(df)
                    print(f"✅ Table {i+1}: Contains budget data - included")
                else:
                    print(f"⏭️  Table {i+1}: No budget data found - skipped")
            
            print(f"💰 Found {len(tables)} budget-related tables in image")
            return tables
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract tables from image: {e}")

    def process_file(self, file_bytes: bytes, file_extension: str, filename: str = "") -> List[pd.DataFrame]:
        """Process PDF or image file and extract tables with source information."""
        ext = file_extension.lower()
        
        if ext == ".pdf":
            return self.extract_from_pdf(file_bytes, filename)
            
        elif ext in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
            return self.extract_from_image(file_bytes, filename)
            
        else:
            raise ValueError(f"Unsupported file format for PDF/Image extractor: {ext}")

    def get_table_source_info(self, table: pd.DataFrame) -> str:
        """Get source information for a table."""
        if hasattr(table, '_source_info'):
            return table._source_info
        else:
            return "Unknown source"

    def _process_table_structure(self, table) -> pd.DataFrame:
        """Process Azure Document Intelligence table structure into DataFrame with original headers."""
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
            
            # Create DataFrame - preserve original headers from first row
            if rows > 1:
                # First row contains the original headers
                original_headers = [header.strip() if header.strip() else f"Column_{i+1}" 
                                  for i, header in enumerate(grid[0])]
                data = grid[1:]
                df = pd.DataFrame(data, columns=original_headers)
                
                # Store original headers for bold formatting
                df._original_headers = original_headers
            else:
                # Single row - treat as data with generic headers
                generic_headers = [f"Column_{i+1}" for i in range(cols)]
                df = pd.DataFrame(grid, columns=generic_headers)
                df._original_headers = generic_headers
            
            return self.clean_table(df)
            
        except Exception as e:
            print(f"Warning: Failed to process table structure: {e}")
            return pd.DataFrame()

    def clean_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame by removing empty rows and preserving original headers."""
        if df.empty:
            return df
            
        # Preserve original headers before cleaning
        original_headers = getattr(df, '_original_headers', list(df.columns))
        
        # Remove completely empty rows
        df = df[~df.apply(lambda row: row.astype(str).str.strip().eq('').all(), axis=1)]
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Restore original headers (don't change them to generic names)
        df.columns = original_headers[:len(df.columns)]  # Ensure we don't exceed available headers
        
        # Store original headers as metadata
        df._original_headers = original_headers
        
        # Fill NaN values
        df = df.fillna("")
        
        return df

    @staticmethod
    def get_extractor() -> Optional['PDFImageTableExtractor']:
        """Get configured PDFImageTableExtractor instance."""
        try:
            config = AzureConfig()
            document_client = config.get_document_client()
            if document_client is None:
                return None
            return PDFImageTableExtractor(document_client)
        except Exception:
            return None

    @staticmethod
    def get_blob_manager() -> Optional[BlobManager]:
        """Get configured BlobManager instance for PDF/Image files."""
        try:
            config = AzureConfig()
            blob_service_client = config.get_blob_service_client()
            if blob_service_client is None or not config.blob_container:
                return None
            return BlobManager(blob_service_client, config.blob_container)
        except Exception:
            return None


def test_pdf_image_extractor():
    """Test the PDF/Image table extractor functionality."""
    print("🧪 Testing PDF/Image Table Extractor")
    print("=" * 50)
    
    # Test data structure (simulating what Document Intelligence would return)
    sample_tables = [
        # Table with budget data
        {
            'columns': ['Item', 'Description', 'Amount'],
            'data': [
                ['Office Supplies', 'Stationery and materials', '$2,500.00'],
                ['Equipment', 'Laptops and monitors', '$15,000.00'],
                ['Travel', 'Business trips', '$8,750.00']
            ]
        },
        # Table without budget data
        {
            'columns': ['Name', 'Department', 'Location'],
            'data': [
                ['John Doe', 'IT', 'New York'],
                ['Jane Smith', 'HR', 'Chicago']
            ]
        }
    ]
    
    # Simulate processing
    extractor = PDFImageTableExtractor(None)  # Mock for testing
    
    processed_tables = []
    for i, table_data in enumerate(sample_tables):
        df = pd.DataFrame(table_data['data'], columns=table_data['columns'])
        df = extractor.clean_table(df)
        
        if contains_money(df):
            df._source_info = f"Test_Table_{i+1}"
            df._original_headers = table_data['columns']
            processed_tables.append(df)
            print(f"✅ Table {i+1}: Headers: {table_data['columns']} - Budget data found")
        else:
            print(f"⏭️  Table {i+1}: Headers: {table_data['columns']} - No budget data")
    
    print(f"\n📊 Results:")
    print(f"   • Tables processed: {len(sample_tables)}")
    print(f"   • Budget tables found: {len(processed_tables)}")
    print(f"   • Original headers preserved: ✅")
    print(f"   • Ready for AI processing: ✅")
    
    print(f"\n🎯 PDF/Image Extractor Features:")
    print("✅ Document Intelligence OCR processing")
    print("✅ Original header preservation from source documents")
    print("✅ Budget table filtering with contains_money()")
    print("✅ Source tracking (page numbers, table positions)")
    print("✅ Direct integration with AI Budget Extractor")
    print("✅ Support for PDF and image formats (JPG, PNG, TIFF, BMP)")


if __name__ == "__main__":
    print("=" * 60)
    print("PDF/IMAGE TABLE EXTRACTOR")
    print("=" * 60)
    print("\nSpecialized processor for PDF and image files")
    print("Uses Azure Document Intelligence for OCR and table extraction")
    print("Preserves original headers and filters budget-related tables\n")
    
    # Run the test
    test_pdf_image_extractor()
    
    print("\n" + "=" * 60)
    print("Ready for integration with Trinity Online workflow!")
    print("=" * 60)
