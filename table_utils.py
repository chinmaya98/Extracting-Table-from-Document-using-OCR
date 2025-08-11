import pandas as pd
import io
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from utils.currency_utils import contains_money
import os
from dotenv import load_dotenv

class AzureConfig:
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

class BlobManager:
    def __init__(self, connection_string, container):
        try:
            self.connection_string = connection_string
            self.container = container
            self.service_client = BlobServiceClient.from_connection_string(connection_string)
            self.container_client = self.service_client.get_container_client(container)
        except Exception as e:
            raise RuntimeError(f"BlobManager initialization failed: {e}")

    def list_files(self, extensions=(".pdf", ".xlsx", ".xls")):
        try:
            return [b.name for b in self.container_client.list_blobs() if b.name.lower().endswith(extensions)]
        except Exception as e:
            raise RuntimeError(f"Failed to list files in Blob Storage: {e}")

    def download_file(self, blob_name):
        try:
            blob_client = self.service_client.get_blob_client(container=self.container, blob=blob_name)
            return blob_client.download_blob().readall()
        except Exception as e:
            raise RuntimeError(f"Failed to download blob '{blob_name}': {e}")

class TableExtractor:
    def __init__(self, endpoint, key):
        try:
            self.client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
        except Exception as e:
            raise RuntimeError(f"TableExtractor initialization failed: {e}")

    def extract_from_pdf(self, pdf_bytes):
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
        return [df for df in tables if contains_money(df)]

    @staticmethod
    def clean_table(df):
        # Remove empty/blank rows
        return df[~df.apply(lambda row: row.astype(str).str.strip().eq('').all(), axis=1)].reset_index(drop=True)
