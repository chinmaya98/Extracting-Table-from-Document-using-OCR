"""
PDF and Image Table Extraction Module
Handles PDF files and image files (JPG, PNG, TIFF) using Azure Document Intelligence.
"""

import io
import os
import re
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
            
            # --- FIX for 'AnalyzeResult' object has no attribute 'words' ---
            # Words are contained within pages, not at the top-level result
            all_words = []
            if result.pages:
                for page in result.pages:
                    if page.words:
                        all_words.extend(page.words)
            # --- END FIX ---

            if result.tables:
                print(f"[DIAGNOSTIC] Azure found {len(result.tables)} table(s) in the document.")
                for table in result.tables:
                    # Pass the list of words to the builder function
                    df = self._build_dataframe_with_confidence(table, all_words)
                    if not df.empty:
                        tables.append(df)
            else:
                print("[DIAGNOSTIC] WARNING: The Azure service did NOT find any tables in this document.")
                    
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
            
            poller = self.client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=image_bytes,
                content_type=kind.mime
            )
            result = poller.result()
            
            tables = []

            # --- FIX for 'AnalyzeResult' object has no attribute 'words' ---
            # Words are contained within pages, not at the top-level result
            all_words = []
            if result.pages:
                for page in result.pages:
                    if page.words:
                        all_words.extend(page.words)
            # --- END FIX ---

            if result.tables:
                for table in result.tables:
                    # Pass the list of words to the builder function
                    df = self._build_dataframe_with_confidence(table, all_words)
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
        file_ext = file_extension.lower()
        
        if file_ext == ".pdf":
            return self.extract_tables_from_pdf(file_bytes)
        elif file_ext in [".jpg", ".jpeg", ".png", ".tiff", ".tif"]:
            try:
                return self.extract_tables_from_image(file_bytes)
            except Exception:
                print("Direct image extraction failed, trying PDF conversion...")
                return self.convert_image_to_pdf_and_extract(file_bytes)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    @staticmethod
    def _build_dataframe_with_confidence(table, words):
        """
        Build a pandas DataFrame from a DocumentIntelligence table.
        This version calculates cell confidence by averaging the confidence 
        of all words found within that cell's spans.
        """
        nrows = table.row_count
        ncols = table.column_count
        # store (value, confidence)
        cells = [[("", 0.0) for _ in range(ncols)] for _ in range(nrows)]
        
        for cell in table.cells:
            cell_content = cell.content
            cell_word_confidences = []

            if cell.spans:
                for span in cell.spans:
                    span_offset_start = span.offset
                    span_offset_end = span.offset + span.length
                    
                    # Find all words that are fully contained within this cell's span
                    for word in words:
                        word_offset_start = word.span.offset
                        word_offset_end = word.span.offset + word.span.length
                        
                        if word_offset_start >= span_offset_start and word_offset_end <= span_offset_end:
                            cell_word_confidences.append(word.confidence)

            # Calculate the average confidence for the cell
            avg_conf = 0.0
            if cell_word_confidences:
                avg_conf = sum(cell_word_confidences) / len(cell_word_confidences)
            
            cells[cell.row_index][cell.column_index] = (cell_content, avg_conf)
        
        # ---
        # FIX: Always use generic headers to avoid misinterpreting page titles.
        # This prevents the garbled column names problem seen in the screenshot.
        headers = [f"Column_{i+1}" for i in range(ncols)]
        data = cells # Use all rows as data
        # ---
        
        df_values = [[v[0] for v in row] for row in data]
        df_conf = [[v[1] for v in row] for row in data]
        
        df = pd.DataFrame(df_values, columns=headers)
        df_conf_df = pd.DataFrame(df_conf, columns=[f"{h}_conf" for h in headers])
        df = pd.concat([df, df_conf_df], axis=1)
        
        df = PDFImageProcessor.clean_table(df)
        
        # Preserve mean cell confidence in DataFrame metadata
        conf_cols = [col for col in df.columns if col.endswith("_conf")]
        if conf_cols and not df[conf_cols].empty:
            df.attrs["confidence_score"] = round(df[conf_cols].mean().mean(), 2)
        else:
            df.attrs["confidence_score"] = 0.0

        return df
    
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
        
        # Identify non-confidence columns for content check
        content_cols = [col for col in df.columns if not col.endswith("_conf")]
        
        # Check if all content columns in a row are empty/whitespace
        if not content_cols: # Handle case where df might only have _conf cols
             return pd.DataFrame()

        is_blank = df[content_cols].apply(
            lambda row: all(str(val).strip() == '' for val in row), axis=1
        )
        
        df_cleaned = df[~is_blank].reset_index(drop=True)
        return df_cleaned.fillna("")
        
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
        metadata = {'total_tables': len(tables), 'table_info': []}
        for i, df in enumerate(tables):
            content_cols = [col for col in df.columns if not col.endswith("_conf")]
            table_info = {
                'table_index': i + 1,
                'rows': len(df),
                'columns': len(content_cols),
                'column_names': content_cols,
                'has_monetary_data': contains_money(df),
                'is_empty': df.empty,
                # This 'confidence_score' is the mean of all cell confidences
                'confidence_score': df.attrs.get("confidence_score", 0.0)
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
            raise ValueError("DOC_INTELLIGENCE_ENDPOINT and DOC_INTELLIGENCE_KEY must be set in .env")
        
        client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
        return PDFImageProcessor(client)
        
    except Exception as e:
        raise RuntimeError(f"Failed to create PDFImageProcessor: {e}")

def print_table_with_confidence(df):
    """Prints the DataFrame, showing value and confidence for each cell."""
    if df.empty:
        print("Table is empty.")
        return

    # Separate data columns from confidence columns
    data_columns = [col for col in df.columns if not col.endswith("_conf")]
    
    # Calculate the mean of all cell confidences for an "overall" score
    conf_cols = [col for col in df.columns if col.endswith("_conf")]
    overall_avg_conf = 0.0
    if conf_cols and not df[conf_cols].empty:
        overall_avg_conf = df[conf_cols].mean().mean()

    print("-" * 100)
    print(f"Table Metadata: Rows={len(df)}, Columns={len(data_columns)}, Mean Cell Confidence={overall_avg_conf:.2f}")
    print("-" * 100)

    # Prepare Headers for printing (using the actual column names)
    header_line = " | ".join(f"{h:<28}" for h in data_columns)
    print(header_line)
    # Separator based on header length
    print("=" * (len(data_columns) * 32)) 

    # Iterate through rows and print content and confidence
    for index, row in df.iterrows():
        row_output = []
        for col in data_columns:
            value = str(row[col])
            # Handle cases where the confidence column might be missing
            confidence = row.get(f"{col}_conf", 0.0)
            
            # Format: Value (Confidence: 0.99)
            truncated_value = value[:15] + "..." if len(value) > 18 else value
            display_text = f"{truncated_value} (Conf: {confidence:.2f})"
            
            row_output.append(f"{display_text:<28}")
        
        print(" | ".join(row_output))
    print("-" * 100)


# Example usage
if __name__ == "__main__":
    # Example of how to use the PDF/Image processor
    try:
        processor = get_pdf_image_processor()
        
        # Example with a PDF file
        with open("sample.pdf", "rb") as f:
            file_bytes = f.read()

        print("--- Analyzing document for tables... ---")
        tables = processor.process_file(file_bytes, ".pdf")
        
        if not tables:
            print("No tables were successfully extracted from the document.")
        else:
            print(f"\nSuccessfully extracted {len(tables)} tables.")
            for i, table_df in enumerate(tables):
                print(f"\n--- Displaying Extracted Table {i+1} ---")
                # Use the helper function that prints per-cell confidence
                print_table_with_confidence(table_df)
                
    except FileNotFoundError:
        print("Error: 'sample.pdf' not found. Please ensure the file is present to test.")
    except Exception as e:
        print(f"Error during processing: {e}")

