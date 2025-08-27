"""
PDF and Image Table Extraction Module
Handles PDF files and image files (JPG, PNG, TIFF) using Azure Document Intelligence.
"""

import io
import os
from PIL import Image
import filetype
import pandas as pd
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from utils.currency_utils import contains_money


class PDFImageProcessor:
    """
    Processes PDF and image files for table extraction using Azure Document Intelligence.
    """
    
    def __init__(self, document_client):
        """
        Initialize with Azure Document Intelligence client.
        
        Args:
            document_client: Azure DocumentIntelligenceClient instance
        """
        self.client = document_client
    
    def extract_tables_from_pdf(self, pdf_bytes):
        """
        Extract tables from a PDF file using Azure Document Intelligence.
        
        Args:
            pdf_bytes: PDF file content as bytes
            
        Returns:
            List of pandas DataFrames containing extracted tables
        """
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
                
                # Use the first row as the header if we have multiple rows
                if nrows > 1:
                    # First row as header, rest as data
                    headers = cells[0]
                    data = cells[1:]
                    df = pd.DataFrame(data, columns=headers)
                else:
                    # Single row - create generic headers
                    headers = [f"Column_{i+1}" for i in range(ncols)]
                    df = pd.DataFrame(cells, columns=headers)
                
                # Clean the table
                df = self.clean_table(df)
                if not df.empty:
                    tables.append(df)
                    
            return tables
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract tables from PDF: {e}")
    
    def extract_tables_from_image(self, image_bytes):
        """
        Extract tables from an image file using Azure Document Intelligence.
        
        Args:
            image_bytes: Image file content as bytes
            
        Returns:
            List of pandas DataFrames containing extracted tables
        """
        try:
            # Detect image type for content_type
            kind = filetype.guess(image_bytes)
            if kind is None or not kind.mime.startswith("image/"):
                raise ValueError("Unsupported or undetectable image type for table extraction.")
            
            content_type = kind.mime
            
            poller = self.client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=image_bytes,
                content_type=content_type
            )
            result = poller.result()
            
            tables = []
            for table in result.tables:
                nrows = table.row_count
                ncols = table.column_count
                cells = [["" for _ in range(ncols)] for _ in range(nrows)]
                
                for cell in table.cells:
                    cells[cell.row_index][cell.column_index] = cell.content
                
                # Use the first row as header if we have multiple rows
                if nrows > 1:
                    headers = cells[0]
                    data = cells[1:]
                    df = pd.DataFrame(data, columns=headers)
                else:
                    # Single row - create generic headers
                    headers = [f"Column_{i+1}" for i in range(ncols)]
                    df = pd.DataFrame(cells, columns=headers)
                
                # Clean the table
                df = self.clean_table(df)
                if not df.empty:
                    tables.append(df)
                    
            return tables
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract tables from image: {e}")
    
    def convert_image_to_pdf_and_extract(self, image_bytes):
        """
        Convert image to PDF format and then extract tables.
        Alternative method for image processing.
        
        Args:
            image_bytes: Image file content as bytes
            
        Returns:
            List of pandas DataFrames containing extracted tables
        """
        try:
            # Convert image to PDF
            image = Image.open(io.BytesIO(image_bytes))
            pdf_buffer = io.BytesIO()
            
            # Convert to RGB if necessary (for PNG with transparency)
            if image.mode in ("RGBA", "LA", "P"):
                rgb_image = Image.new("RGB", image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
                image = rgb_image
            
            image.save(pdf_buffer, format="PDF")
            pdf_buffer.seek(0)
            
            # Extract tables from the converted PDF
            return self.extract_tables_from_pdf(pdf_buffer.read())
            
        except Exception as e:
            raise RuntimeError(f"Failed to convert image to PDF and extract tables: {e}")
    
    def process_file(self, file_bytes, file_extension):
        """
        Process a PDF or image file and extract tables.
        
        Args:
            file_bytes: File content as bytes
            file_extension: File extension (e.g., '.pdf', '.jpg', '.png')
            
        Returns:
            List of pandas DataFrames containing extracted tables
        """
        try:
            file_ext = file_extension.lower()
            
            if file_ext == ".pdf":
                return self.extract_tables_from_pdf(file_bytes)
            elif file_ext in [".jpg", ".jpeg", ".png", ".tiff", ".tif"]:
                # Try direct image extraction first
                try:
                    return self.extract_tables_from_image(file_bytes)
                except Exception:
                    # Fallback to image-to-PDF conversion
                    print("Direct image extraction failed, trying PDF conversion...")
                    return self.convert_image_to_pdf_and_extract(file_bytes)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}. "
                               "Supported formats: .pdf, .jpg, .jpeg, .png, .tiff, .tif")
                
        except Exception as e:
            raise RuntimeError(f"Failed to process file with extension {file_extension}: {e}")
    
    @staticmethod
    def clean_table(df):
        """
        Remove empty/blank rows from a DataFrame and clean up data.
        
        Args:
            df: pandas DataFrame to clean
            
        Returns:
            Cleaned pandas DataFrame
        """
        if df.empty:
            return df
        
        # Remove completely empty rows
        def is_empty_row(row):
            return all(str(val).strip() == '' for val in row)
        
        df_cleaned = df[~df.apply(is_empty_row, axis=1)]
        
        # Reset index
        df_cleaned = df_cleaned.reset_index(drop=True)
        
        # Fill NaN values with empty strings
        df_cleaned = df_cleaned.fillna("")
        
        return df_cleaned
    
    @staticmethod
    def filter_budget_tables(tables):
        """
        Filter tables to only those containing monetary values.
        
        Args:
            tables: List of pandas DataFrames
            
        Returns:
            List of DataFrames that contain monetary values
        """
        budget_tables = []
        for df in tables:
            if contains_money(df):
                budget_tables.append(df)
        return budget_tables
    
    def get_table_metadata(self, tables):
        """
        Get metadata information about extracted tables.
        
        Args:
            tables: List of pandas DataFrames
            
        Returns:
            Dictionary with table metadata
        """
        metadata = {
            'total_tables': len(tables),
            'table_info': []
        }
        
        for i, df in enumerate(tables):
            table_info = {
                'table_index': i + 1,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'has_monetary_data': contains_money(df),
                'is_empty': df.empty
            }
            metadata['table_info'].append(table_info)
        
        return metadata


def get_pdf_image_processor():
    """
    Factory function to create a PDFImageProcessor instance with Azure configuration.
    
    Returns:
        PDFImageProcessor instance configured with Azure Document Intelligence
    """
    try:
        load_dotenv()
        endpoint = os.getenv("DOC_INTELLIGENCE_ENDPOINT")
        key = os.getenv("DOC_INTELLIGENCE_KEY")
        
        if not all([endpoint, key]):
            raise ValueError("DOC_INTELLIGENCE_ENDPOINT and DOC_INTELLIGENCE_KEY must be set in environment variables")
        
        document_client = DocumentIntelligenceClient(
            endpoint=endpoint, 
            credential=AzureKeyCredential(key)
        )
        
        return PDFImageProcessor(document_client)
        
    except Exception as e:
        raise RuntimeError(f"Failed to create PDFImageProcessor: {e}")


# Example usage
if __name__ == "__main__":
    # Example of how to use the PDF/Image processor
    try:
        processor = get_pdf_image_processor()
        
        # Example with a PDF file
        with open("sample.pdf", "rb") as f:
            pdf_bytes = f.read()
        
        tables = processor.process_file(pdf_bytes, ".pdf")
        metadata = processor.get_table_metadata(tables)
        
        print(f"Extracted {metadata['total_tables']} tables from PDF")
        for info in metadata['table_info']:
            print(f"Table {info['table_index']}: {info['rows']} rows, {info['columns']} columns")
            
    except Exception as e:
        print(f"Error: {e}")
