import pandas as pd
import io
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from utils.currency_utils import contains_money
import os
from dotenv import load_dotenv

class AzureConfig:
    """
    Loads Azure configuration from environment variables.
    Raises RuntimeError if any required variable is missing.
    """
    def __init__(self):
        try:
            load_dotenv()
            self.endpoint = os.getenv("DOC_INTELLIGENCE_ENDPOINT")
            self.key = os.getenv("DOC_INTELLIGENCE_KEY")
            self.blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
            self.blob_container = os.getenv("AZURE_BLOB_CONTAINER")
            if not all([self.endpoint, self.key, self.blob_connection_string, self.blob_container]):
                raise ValueError("One or more Azure environment variables are missing.")
        except Exception as e:
            raise RuntimeError(f"AzureConfig initialization failed: {e}")

    def get_document_client(self):
        """Returns a DocumentIntelligenceClient instance."""
        try:
            return DocumentIntelligenceClient(endpoint=self.endpoint, credential=AzureKeyCredential(self.key))
        except Exception as e:
            raise RuntimeError(f"Failed to create DocumentIntelligenceClient: {e}")

    def get_blob_service_client(self):
        """Returns a BlobServiceClient instance."""
        try:
            return BlobServiceClient.from_connection_string(self.blob_connection_string)
        except Exception as e:
            raise RuntimeError(f"Failed to create BlobServiceClient: {e}")

class BlobManager:
    """
    Manages Azure Blob Storage operations.
    """
    def __init__(self, service_client, container):
        try:
            self.service_client = service_client
            self.container = container
            self.container_client = self.service_client.get_container_client(container)
        except Exception as e:
            raise RuntimeError(f"BlobManager initialization failed: {e}")

    def list_files(self, extensions=(".pdf", ".xlsx", ".xls")):
        """Lists files in the container with given extensions."""
        try:
            return [b.name for b in self.container_client.list_blobs() if b.name.lower().endswith(extensions)]
        except Exception as e:
            raise RuntimeError(f"Failed to list files in Blob Storage: {e}")

    def download_file(self, blob_name):
        """Downloads a blob and returns its bytes."""
        try:
            blob_client = self.service_client.get_blob_client(container=self.container, blob=blob_name)
            return blob_client.download_blob().readall()
        except Exception as e:
            raise RuntimeError(f"Failed to download blob '{blob_name}': {e}")

class TableExtractor:
    """
    Extracts tables from PDF and Excel files using Azure Document Intelligence and pandas.
    """
    def __init__(self, document_client):
        self.client = document_client

    def extract_from_pdf(self, pdf_bytes):
        """Extracts tables from a PDF file using Azure Document Intelligence."""
        try:
            poller = self.client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=pdf_bytes,
                content_type="application/pdf"
            )
            result = poller.result()
            tables = []
            for table in result.tables:
                nrows = table.row_count
                ncols = table.column_count
                cells = [["" for _ in range(ncols)] for _ in range(nrows)]
                for cell in table.cells:
                    cells[cell.row_index][cell.column_index] = cell.content
                df = pd.DataFrame(cells)
                tables.append(df)
            return tables
        except Exception as e:
            raise RuntimeError(f"Failed to extract tables from PDF: {e}")

    def extract_from_excel(self, file_bytes):
        """Extracts tables from an Excel file using pandas."""
        try:
            excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
            tables = []
            for sheet in excel_file.sheet_names:
                df = excel_file.parse(sheet)
                if not df.empty:
                    tables.append(df)
            return tables
        except Exception as e:
            raise RuntimeError(f"Failed to extract tables from Excel: {e}")

    def process_file(self, file_bytes, file_ext):
        """Processes a file and extracts tables based on its extension."""
        try:
            if file_ext == ".pdf":
                return self.extract_from_pdf(file_bytes)
            elif file_ext in [".xlsx", ".xls"]:
                return self.extract_from_excel(file_bytes)
            else:
                return []
        except Exception as e:
            raise RuntimeError(f"Failed to process file: {e}")

    @staticmethod
    def filter_tables(tables):
        """Filters tables to only those containing monetary values."""
        return [df for df in tables if contains_money(df)]

    @staticmethod
    def clean_table(df):
        """Removes empty/blank rows from a DataFrame."""
        return df[~df.apply(lambda row: row.astype(str).str.strip().eq('').all(), axis=1)].reset_index(drop=True)

def get_blob_manager_and_extractor():
    """
    Utility function to get initialized BlobManager and TableExtractor instances.
    Returns (blob_manager, extractor) tuple.
    """
    config = AzureConfig()
    document_client = config.get_document_client()
    blob_service_client = config.get_blob_service_client()
    blob_manager = BlobManager(blob_service_client, config.blob_container)
    extractor = TableExtractor(document_client)
    return blob_manager, extractor
