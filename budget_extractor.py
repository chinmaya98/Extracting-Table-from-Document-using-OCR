import pandas as pd
import io
import re
import numpy as np
import streamlit as st
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from table_utils import AzureConfig, BlobManager, TableExtractor
from utils.currency_utils import MONEY_PATTERN, MONEY_KEYWORDS
import os
from dotenv import load_dotenv

class BudgetExtractor:
    """
    Comprehensive budget and label extraction from PDFs, images, and Excel files.
    Extracts maximum 3 columns: Label1, Label2, Budget/Amount
    """
    
    def __init__(self):
        load_dotenv()
        self.config = AzureConfig()
        self.document_client = self.config.get_document_client()
        self.blob_service_client = self.config.get_blob_service_client()
        self.blob_manager = BlobManager(self.blob_service_client, self.config.blob_container)
        self.table_extractor = TableExtractor(self.document_client)
        
        # Enhanced patterns for budget detection
        self.currency_symbols = ['$', '€', '£', '¥', '₹', '₩', '₽', 'Rs', 'USD', 'EUR', 'GBP', 'INR']
        self.budget_keywords = [
            'budget', 'amount', 'cost', 'price', 'total', 'value', 'expense', 
            'revenue', 'income', 'salary', 'wage', 'fee', 'payment', 'sum',
            'balance', 'fund', 'allocation', 'expenditure', 'spend', 'investment'
        ]
        
        # Enhanced money pattern with more coverage
        self.enhanced_money_pattern = re.compile(
            r'(?:'
            r'(?:\$|€|£|¥|₹|₩|₽|Rs\.?|USD|EUR|GBP|INR)\s*[\d,]+(?:\.\d{1,2})?|'  # Symbol before amount
            r'[\d,]+(?:\.\d{1,2})?\s*(?:\$|€|£|¥|₹|₩|₽|Rs\.?|USD|EUR|GBP|INR)|'  # Amount before symbol
            r'[\d,]+(?:\.\d{1,2})?(?:\s*(?:dollars?|euros?|pounds?|rupees?|yen))|'  # Amount with word
            r'(?:Rs\.?|INR)\s*[\d,]+(?:\.\d{1,2})?|'  # Indian Rupee specific
            r'[\d,]+(?:\.\d{1,2})?\s*(?:K|k|M|m|B|b|thousand|million|billion)'  # Abbreviated amounts
            r')',
            re.IGNORECASE
        )

    def extract_budget_data(self, file_path: str = None, file_bytes: bytes = None, file_extension: str = None) -> Dict[str, Any]:
        """
        Main method to extract budget data from various file formats.
        Returns a dictionary with extracted data and metadata.
        """
        try:
            if file_path:
                # Download from blob storage
                file_bytes = self.blob_manager.download_file(file_path)
                file_extension = os.path.splitext(file_path)[1].lower()
            
            if not file_bytes or not file_extension:
                raise ValueError("Either file_path or both file_bytes and file_extension must be provided")
            
            # Process based on file type
            if file_extension in ['.pdf']:
                return self._extract_from_pdf(file_bytes, file_path or "PDF Document")
            elif file_extension in ['.xlsx', '.xls']:
                return self._extract_from_excel(file_bytes, file_path or "Excel Document")
            elif file_extension in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                return self._extract_from_image(file_bytes, file_path or "Image Document")
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'file_name': file_path or "Unknown",
                'extracted_data': pd.DataFrame()
            }

    def _extract_from_pdf(self, pdf_bytes: bytes, file_name: str) -> Dict[str, Any]:
        """Extract budget data from PDF files using OCR and table detection."""
        try:
            # Use Azure Document Intelligence for OCR and table extraction
            poller = self.document_client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=pdf_bytes,
                content_type="application/pdf"
            )
            result = poller.result()
            
            all_budget_data = []
            
            # Extract from tables
            for table_idx, table in enumerate(result.tables):
                table_data = self._process_table_structure(table, f"PDF_Table_{table_idx}")
                if not table_data.empty:
                    all_budget_data.append(table_data)
            
            # Extract from plain text (OCR data)
            if result.content:
                text_data = self._extract_from_text(result.content, "PDF_Text")
                if not text_data.empty:
                    all_budget_data.append(text_data)
            
            # Combine all extracted data
            final_data = self._combine_and_process_data(all_budget_data)
            
            return {
                'success': True,
                'file_name': file_name,
                'file_type': 'PDF',
                'extracted_data': final_data,
                'highest_budget_row': self._find_highest_budget_row(final_data),
                'total_rows': len(final_data),
                'budget_columns_found': self._get_budget_columns(final_data)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'file_name': file_name,
                'extracted_data': pd.DataFrame()
            }

    def _extract_from_excel(self, excel_bytes: bytes, file_name: str) -> Dict[str, Any]:
        """Extract budget data from Excel files."""
        try:
            excel_file = pd.ExcelFile(io.BytesIO(excel_bytes))
            all_budget_data = []
            
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    if df.empty:
                        continue
                    
                    # Clean and process the sheet
                    processed_data = self._process_excel_sheet(df, f"Excel_{sheet_name}")
                    if not processed_data.empty:
                        all_budget_data.append(processed_data)
                        
                except Exception as sheet_error:
                    print(f"Error processing sheet {sheet_name}: {sheet_error}")
                    continue
            
            # Combine all sheets data
            final_data = self._combine_and_process_data(all_budget_data)
            
            return {
                'success': True,
                'file_name': file_name,
                'file_type': 'Excel',
                'extracted_data': final_data,
                'highest_budget_row': self._find_highest_budget_row(final_data),
                'total_rows': len(final_data),
                'sheets_processed': len(excel_file.sheet_names),
                'budget_columns_found': self._get_budget_columns(final_data)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'file_name': file_name,
                'extracted_data': pd.DataFrame()
            }

    def _extract_from_image(self, image_bytes: bytes, file_name: str) -> Dict[str, Any]:
        """Extract budget data from image files using OCR."""
        try:
            # Convert image to PDF for Azure Document Intelligence
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save as PDF for processing
            pdf_bytes = io.BytesIO()
            image.save(pdf_bytes, format="PDF", quality=95)
            pdf_bytes.seek(0)
            
            # Process as PDF
            return self._extract_from_pdf(pdf_bytes.read(), file_name)
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'file_name': file_name,
                'extracted_data': pd.DataFrame()
            }

    def _process_table_structure(self, table, source_name: str) -> pd.DataFrame:
        """Process Azure Document Intelligence table structure."""
        try:
            nrows = table.row_count
            ncols = table.column_count
            cells = [["" for _ in range(ncols)] for _ in range(nrows)]
            
            # Fill the cells
            for cell in table.cells:
                cells[cell.row_index][cell.column_index] = str(cell.content).strip()
            
            # Create DataFrame
            if nrows > 1:
                headers = [self._clean_header(cell) for cell in cells[0]]
                df = pd.DataFrame(cells[1:], columns=headers)
            else:
                df = pd.DataFrame(cells)
            
            # Process the DataFrame to extract budget information
            return self._extract_budget_from_dataframe(df, source_name)
            
        except Exception as e:
            print(f"Error processing table structure: {e}")
            return pd.DataFrame()

    def _process_excel_sheet(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Process Excel sheet to extract budget information."""
        try:
            # Clean the DataFrame
            df = df.dropna(how='all').reset_index(drop=True)  # Remove completely empty rows
            df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns
            
            # Fill NaN values
            df = df.fillna('')
            
            # Extract budget information
            return self._extract_budget_from_dataframe(df, source_name)
            
        except Exception as e:
            print(f"Error processing Excel sheet: {e}")
            return pd.DataFrame()

    def _extract_from_text(self, text: str, source_name: str) -> pd.DataFrame:
        """Extract budget information from plain text using regex patterns."""
        try:
            lines = text.split('\n')
            extracted_data = []
            
            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Find money amounts in the line
                money_matches = list(self.enhanced_money_pattern.finditer(line))
                
                if money_matches:
                    for match in money_matches:
                        amount_str = match.group()
                        amount_value = self._parse_amount_value(amount_str)
                        
                        # Extract context around the amount as labels
                        before_amount = line[:match.start()].strip()
                        after_amount = line[match.end():].strip()
                        
                        # Create labels from context
                        label1 = self._extract_meaningful_label(before_amount)
                        label2 = self._extract_meaningful_label(after_amount)
                        
                        if label1 or label2:  # Only add if we have some context
                            currency_symbol = self._extract_currency_symbol(amount_str)
                            
                            entry = {
                                'Label1': label1,
                                'Budget_Amount': amount_value,
                                'Currency_Symbol': currency_symbol,
                                'Original_Text': amount_str,
                                'Source': source_name,
                                'Line_Number': line_idx + 1
                            }
                            
                            # Only add Label2 if we have meaningful content
                            if label2:
                                entry['Label2'] = label2
                            
                            extracted_data.append(entry)
            
            return pd.DataFrame(extracted_data) if extracted_data else pd.DataFrame()
            
        except Exception as e:
            print(f"Error extracting from text: {e}")
            return pd.DataFrame()

    def _extract_budget_from_dataframe(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Extract budget information from a DataFrame and standardize to 2-3 columns with smart budget selection."""
        try:
            if df.empty:
                return pd.DataFrame()
            
            # Convert all data to string for processing
            df = df.astype(str)
            
            # Find all potential budget columns
            budget_columns = self._find_budget_columns(df)
            
            if not budget_columns:
                return pd.DataFrame()
            
            extracted_data = []
            
            for _, row in df.iterrows():
                # Find the highest budget amount across all budget columns
                max_amount = 0
                best_budget_info = None
                
                for budget_col in budget_columns:
                    amount_str = str(row[budget_col]).strip()
                    if not amount_str or amount_str.lower() in ['', 'nan', 'none']:
                        continue
                    
                    # Check if this looks like a monetary value
                    if self.enhanced_money_pattern.search(amount_str) or self._is_numeric_amount(amount_str):
                        amount_value = self._parse_amount_value(amount_str)
                        
                        # Keep track of the highest amount and its details
                        if amount_value > max_amount:
                            max_amount = amount_value
                            currency_symbol = self._extract_currency_symbol(amount_str)
                            best_budget_info = {
                                'amount': amount_value,
                                'original_text': amount_str,
                                'currency_symbol': currency_symbol,
                                'column': budget_col
                            }
                
                # If we found a valid budget amount, create the entry
                if best_budget_info and best_budget_info['amount'] > 0:
                    # Get other columns as labels (exclude the budget column)
                    other_columns = [col for col in df.columns if col != best_budget_info['column']]
                    
                    # Create labels from other columns
                    labels = []
                    for col in other_columns:
                        label_value = str(row[col]).strip()
                        if label_value and label_value.lower() not in ['', 'nan', 'none']:
                            labels.append(label_value)
                        if len(labels) >= 2:  # Limit to 2 labels max
                            break
                    
                    # Create the entry - only include Label2 if we have meaningful content
                    entry = {
                        'Label1': labels[0] if len(labels) > 0 else best_budget_info['column'],
                        'Budget_Amount': best_budget_info['amount'],
                        'Currency_Symbol': best_budget_info['currency_symbol'],
                        'Original_Text': best_budget_info['original_text'],
                        'Source': source_name,
                        'Budget_Column': best_budget_info['column']
                    }
                    
                    # Only add Label2 if we have a meaningful second label
                    if len(labels) > 1 and labels[1]:
                        entry['Label2'] = labels[1]
                    
                    extracted_data.append(entry)
            
            return pd.DataFrame(extracted_data) if extracted_data else pd.DataFrame()
            
        except Exception as e:
            print(f"Error extracting budget from DataFrame: {e}")
            return pd.DataFrame()

    def _find_budget_columns(self, df: pd.DataFrame) -> List[str]:
        """Find columns that likely contain budget/amount information."""
        budget_columns = []
        
        for col in df.columns:
            col_lower = str(col).lower()
            
            # Check column name for budget keywords
            if any(keyword in col_lower for keyword in self.budget_keywords):
                budget_columns.append(col)
                continue
            
            # Check column content for money patterns
            money_count = 0
            total_non_empty = 0
            
            for value in df[col].astype(str):
                if value and value.lower() not in ['', 'nan', 'none']:
                    total_non_empty += 1
                    if (self.enhanced_money_pattern.search(value) or 
                        self._is_numeric_amount(value)):
                        money_count += 1
            
            # If more than 30% of non-empty values look like money, consider it a budget column
            if total_non_empty > 0 and (money_count / total_non_empty) > 0.3:
                budget_columns.append(col)
        
        return budget_columns

    def _is_numeric_amount(self, value: str) -> bool:
        """Check if a string represents a numeric amount that could be money."""
        try:
            # Remove common separators and check if it's a number
            cleaned = re.sub(r'[,\s]', '', str(value).strip())
            if cleaned:
                float(cleaned)
                # Additional check: should be a reasonable monetary amount
                return len(cleaned) >= 1 and not cleaned.startswith('0000')
        except (ValueError, TypeError):
            pass
        return False

    def _extract_currency_symbol(self, amount_str: str) -> str:
        """Extract currency symbol from amount string."""
        try:
            # Common currency symbols and their text representations
            currency_map = {
                '$': '$', 'USD': '$', 'usd': '$', 'dollar': '$', 'dollars': '$',
                '€': '€', 'EUR': '€', 'eur': '€', 'euro': '€', 'euros': '€',
                '£': '£', 'GBP': '£', 'gbp': '£', 'pound': '£', 'pounds': '£',
                '₹': '₹', 'INR': '₹', 'inr': '₹', 'Rs': '₹', 'rs': '₹', 'rupee': '₹', 'rupees': '₹',
                '¥': '¥', 'JPY': '¥', 'jpy': '¥', 'yen': '¥',
                '₩': '₩', 'KRW': '₩', 'krw': '₩', 'won': '₩',
                '₽': '₽', 'RUB': '₽', 'rub': '₽', 'ruble': '₽', 'rubles': '₽'
            }
            
            # Check for direct symbol matches first
            for symbol, standard_symbol in currency_map.items():
                if symbol in amount_str:
                    return standard_symbol
            
            # If no symbol found, default to $ (most common)
            return '$'
            
        except Exception:
            return '$'

    def _parse_amount_value(self, amount_str: str) -> float:
        """Parse amount string to numeric value."""
        try:
            # Remove currency symbols and clean up
            cleaned = re.sub(r'[^\d.,KkMmBb]', '', amount_str)
            
            # Handle abbreviations
            multiplier = 1
            if re.search(r'[Kk]', cleaned):
                multiplier = 1000
                cleaned = re.sub(r'[Kk]', '', cleaned)
            elif re.search(r'[Mm]', cleaned):
                multiplier = 1000000
                cleaned = re.sub(r'[Mm]', '', cleaned)
            elif re.search(r'[Bb]', cleaned):
                multiplier = 1000000000
                cleaned = re.sub(r'[Bb]', '', cleaned)
            
            # Remove extra commas and convert
            cleaned = re.sub(r',(?=\d{3})', '', cleaned)  # Remove thousand separators
            
            if cleaned:
                return float(cleaned) * multiplier
            return 0.0
        except (ValueError, TypeError):
            return 0.0

    def _extract_meaningful_label(self, text: str) -> str:
        """Extract meaningful label from text context."""
        if not text:
            return ''
        
        # Remove common noise words and clean up
        noise_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word.strip() for word in text.split() if word.strip().lower() not in noise_words]
        
        # Take meaningful words (limit to reasonable length)
        meaningful_words = []
        for word in words:
            if len(word) > 1 and not word.isdigit():
                meaningful_words.append(word)
            if len(meaningful_words) >= 3:  # Limit label length
                break
        
        return ' '.join(meaningful_words)

    def _clean_header(self, header: str) -> str:
        """Clean column header."""
        if not header:
            return 'Unknown'
        return str(header).strip().replace('\n', ' ').replace('\r', '')

    def _combine_and_process_data(self, data_list: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple DataFrames and process to final format with dynamic columns."""
        if not data_list:
            return pd.DataFrame(columns=['Label1', 'Budget_Amount'])
        
        # Combine all DataFrames
        combined_df = pd.concat(data_list, ignore_index=True)
        
        if combined_df.empty:
            return pd.DataFrame(columns=['Label1', 'Budget_Amount'])
        
        # Ensure required columns exist
        if 'Label1' not in combined_df.columns:
            combined_df['Label1'] = ''
        if 'Budget_Amount' not in combined_df.columns:
            combined_df['Budget_Amount'] = 0.0
        if 'Currency_Symbol' not in combined_df.columns:
            combined_df['Currency_Symbol'] = '$'
        
        # Clean and standardize
        combined_df['Label1'] = combined_df['Label1'].astype(str).str.strip()
        combined_df['Budget_Amount'] = pd.to_numeric(combined_df['Budget_Amount'], errors='coerce').fillna(0)
        combined_df['Currency_Symbol'] = combined_df['Currency_Symbol'].astype(str).str.strip()
        
        # Handle Label2 if it exists and has meaningful content
        if 'Label2' in combined_df.columns:
            combined_df['Label2'] = combined_df['Label2'].astype(str).str.strip()
            # Remove empty Label2 values
            combined_df.loc[combined_df['Label2'].isin(['', 'nan', 'None']), 'Label2'] = None
        
        # Remove rows with no meaningful data
        valid_rows = (
            (combined_df['Label1'] != '') | 
            (combined_df['Budget_Amount'] > 0)
        )
        
        result_df = combined_df[valid_rows].copy()
        
        # Sort by budget amount (highest first)
        if not result_df.empty:
            result_df = result_df.sort_values('Budget_Amount', ascending=False).reset_index(drop=True)
        
        # Determine final column structure - Budget_Amount should always be last
        final_columns = ['Label1']
        
        # Add Label2 only if it contains meaningful data
        if 'Label2' in result_df.columns and result_df['Label2'].notna().any():
            final_columns.append('Label2')
        
        # Always add Budget_Amount and Currency_Symbol as the last columns
        final_columns.extend(['Budget_Amount', 'Currency_Symbol'])
        
        # Return with dynamic column structure
        return result_df[final_columns].copy()

    def _find_highest_budget_row(self, df: pd.DataFrame) -> Optional[Dict]:
        """Find the row with the highest budget amount."""
        if df.empty or 'Budget_Amount' not in df.columns:
            return None
        
        max_idx = df['Budget_Amount'].idxmax()
        max_row = df.loc[max_idx]
        
        result = {
            'index': max_idx,
            'label1': max_row['Label1'],
            'budget_amount': max_row['Budget_Amount'],
            'row_data': max_row.to_dict()
        }
        
        # Only add label2 if it exists and has content
        if 'Label2' in df.columns and pd.notna(max_row.get('Label2')) and max_row.get('Label2'):
            result['label2'] = max_row['Label2']
        
        return result

    def _get_budget_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of columns that contain budget information."""
        if df.empty:
            return []
        return ['Budget_Amount']  # Our standardized budget column

    def process_multiple_files(self, file_list: List[str] = None) -> Dict[str, Any]:
        """Process multiple files and return combined results."""
        if file_list is None:
            # Get all supported files from blob storage
            file_list = self.blob_manager.list_files(extensions=(".pdf", ".xlsx", ".xls", ".jpg", ".jpeg", ".png", ".tiff"))
        
        results = {}
        all_data = []
        
        for file_name in file_list:
            print(f"Processing: {file_name}")
            result = self.extract_budget_data(file_path=file_name)
            results[file_name] = result
            
            if result['success'] and not result['extracted_data'].empty:
                # Add file name to each row
                file_data = result['extracted_data'].copy()
                file_data['Source_File'] = file_name
                all_data.append(file_data)
        
        # Combine all data
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.sort_values('Budget_Amount', ascending=False).reset_index(drop=True)
        else:
            combined_data = pd.DataFrame(columns=['Label1', 'Label2', 'Budget_Amount', 'Source_File'])
        
        return {
            'individual_results': results,
            'combined_data': combined_data,
            'total_files_processed': len(file_list),
            'successful_extractions': sum(1 for r in results.values() if r['success']),
            'total_budget_entries': len(combined_data),
            'highest_budget_overall': self._find_highest_budget_row(combined_data)
        }

# Utility functions for easy access
def extract_budget_from_file(file_path: str) -> Dict[str, Any]:
    """
    Convenient function to extract budget data from a single file.
    """
    extractor = BudgetExtractor()
    return extractor.extract_budget_data(file_path=file_path)

def extract_budget_from_all_files() -> Dict[str, Any]:
    """
    Convenient function to extract budget data from all supported files in blob storage.
    """
    extractor = BudgetExtractor()
    return extractor.process_multiple_files()

def get_budget_summary(file_path: str = None) -> pd.DataFrame:
    """
    Get a clean summary of budget data with dynamic column structure.
    Budget_Amount is always the last column, Label2 only if present.
    If file_path is None, processes all files.
    """
    extractor = BudgetExtractor()
    
    if file_path:
        result = extractor.extract_budget_data(file_path=file_path)
        return result.get('extracted_data', pd.DataFrame())
    else:
        result = extractor.process_multiple_files()
        return result.get('combined_data', pd.DataFrame())

# Streamlit UI Functions
def handle_budget_extraction(selected_file):
    """Handle budget extraction functionality in Streamlit."""
    st.header("💰 Budget & Label Extraction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Extract Budget from Selected File"):
            with st.spinner(f"Extracting budget data from {selected_file}..."):
                result = extract_budget_from_file(selected_file)
                display_budget_results(result, single_file=True)
    
    with col2:
        if st.button("Extract Budget from All Files"):
            with st.spinner("Processing all files for budget extraction..."):
                result = extract_budget_from_all_files()
                display_budget_results(result, single_file=False)

def display_budget_results(result, single_file=True):
    """Display budget extraction results in Streamlit."""
    if single_file:
        # Single file results
        if result.get('success', False):
            st.success(f"✅ Successfully processed: {result['file_name']}")
            
            extracted_data = result.get('extracted_data')
            if not extracted_data.empty:
                st.subheader("📊 Extracted Budget Data")
                st.write(f"**Total entries found:** {len(extracted_data)}")
                
                # Display with dynamic columns - Budget_Amount should be last
                display_columns = [col for col in extracted_data.columns 
                                 if col in ['Label1', 'Label2', 'Budget_Amount', 'Currency_Symbol']]
                
                # Create display DataFrame with formatted amounts
                display_df = extracted_data[display_columns].copy()
                
                # Format the budget column with currency symbol
                if 'Budget_Amount' in display_df.columns and 'Currency_Symbol' in display_df.columns:
                    display_df['Formatted_Budget'] = display_df.apply(
                        lambda row: f"{row['Currency_Symbol']}{row['Budget_Amount']:,.2f}", axis=1
                    )
                    # Replace Budget_Amount with formatted version
                    display_df = display_df.drop(['Budget_Amount', 'Currency_Symbol'], axis=1)
                    display_df = display_df.rename(columns={'Formatted_Budget': 'Budget_Amount'})
                
                st.dataframe(display_df, use_container_width=True)
                
                # Highlight highest budget
                highest_budget = result.get('highest_budget_row')
                if highest_budget:
                    st.subheader("🏆 Highest Budget Entry")
                    currency_symbol = extracted_data.loc[highest_budget['index'], 'Currency_Symbol'] if 'Currency_Symbol' in extracted_data.columns else '$'
                    
                    info_text = f"**Label 1:** {highest_budget['label1']}\n\n"
                    
                    # Only show Label 2 if it exists and has content
                    if 'Label2' in extracted_data.columns and extracted_data.loc[highest_budget['index'], 'Label2']:
                        info_text += f"**Label 2:** {highest_budget['label2']}\n\n"
                    
                    info_text += f"**Amount:** {currency_symbol}{highest_budget['budget_amount']:,.2f}"
                    
                    st.info(info_text)
                
                # Show additional metadata
                if st.expander("📋 Processing Details"):
                    st.write(f"**File Type:** {result.get('file_type', 'Unknown')}")
                    st.write(f"**Budget Columns Found:** {result.get('budget_columns_found', [])}")
                    if 'sheets_processed' in result:
                        st.write(f"**Excel Sheets Processed:** {result['sheets_processed']}")
                
                # Download option
                csv_data = extracted_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name=f"budget_data_{result['file_name'].replace('.', '_')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No budget data found in the selected file.")
        else:
            st.error(f"❌ Failed to process file: {result.get('error', 'Unknown error')}")
    
    else:
        # Multiple files results
        individual_results = result.get('individual_results', {})
        combined_data = result.get('combined_data')
        
        st.success(f"✅ Processed {result.get('total_files_processed', 0)} files")
        st.write(f"**Successful extractions:** {result.get('successful_extractions', 0)}")
        st.write(f"**Total budget entries found:** {result.get('total_budget_entries', 0)}")
        
        if not combined_data.empty:
            st.subheader("📊 Combined Budget Data from All Files")
            
            # Show top 10 highest budgets with dynamic columns
            st.write("**Top 10 Highest Budget Entries:**")
            top_entries = combined_data.head(10)
            
            # Create display DataFrame with formatted amounts
            display_columns = [col for col in top_entries.columns 
                             if col in ['Label1', 'Label2', 'Budget_Amount', 'Currency_Symbol', 'Source_File']]
            display_df = top_entries[display_columns].copy()
            
            # Format the budget column with currency symbol
            if 'Budget_Amount' in display_df.columns and 'Currency_Symbol' in display_df.columns:
                display_df['Formatted_Budget'] = display_df.apply(
                    lambda row: f"{row['Currency_Symbol']}{row['Budget_Amount']:,.2f}", axis=1
                )
                # Replace Budget_Amount with formatted version and remove Currency_Symbol
                display_df = display_df.drop(['Budget_Amount', 'Currency_Symbol'], axis=1)
                display_df = display_df.rename(columns={'Formatted_Budget': 'Budget_Amount'})
            
            st.dataframe(display_df, use_container_width=True)
            
            # Overall highest budget
            highest_overall = result.get('highest_budget_overall')
            if highest_overall:
                st.subheader("🏆 Overall Highest Budget Entry")
                source_file = combined_data.loc[highest_overall['index'], 'Source_File']
                currency_symbol = combined_data.loc[highest_overall['index'], 'Currency_Symbol'] if 'Currency_Symbol' in combined_data.columns else '$'
                
                info_text = f"**Label 1:** {highest_overall['label1']}\n\n"
                
                # Only show Label 2 if it exists and has content
                if 'Label2' in combined_data.columns and combined_data.loc[highest_overall['index'], 'Label2']:
                    info_text += f"**Label 2:** {highest_overall['label2']}\n\n"
                
                info_text += f"**Amount:** {currency_symbol}{highest_overall['budget_amount']:,.2f}\n\n"
                info_text += f"**Source File:** {source_file}"
                
                st.info(info_text)
            
            # File-by-file summary
            if st.expander("📁 File-by-File Summary"):
                for file_name, file_result in individual_results.items():
                    if file_result.get('success', False):
                        data_count = len(file_result.get('extracted_data', []))
                        st.write(f"**{file_name}:** {data_count} budget entries found")
                    else:
                        st.write(f"**{file_name}:** ❌ Failed - {file_result.get('error', 'Unknown error')}")
            
            # Download combined results
            csv_data = combined_data.to_csv(index=False)
            st.download_button(
                label="📥 Download All Results as CSV",
                data=csv_data,
                file_name="combined_budget_data.csv",
                mime="text/csv"
            )
        else:
            st.warning("No budget data found in any of the processed files.")
