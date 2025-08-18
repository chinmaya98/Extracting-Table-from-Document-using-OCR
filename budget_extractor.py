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
    Extracts maximum 3 columns: Primary Label, Secondary Label, Budget/Amount
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
        
        # Keywords for identifying primary description columns (Label1)
        self.primary_label_keywords = [
            'service', 'description', 'item', 'product', 'name', 'title', 
            'project', 'task', 'activity', 'work', 'job', 'category',
            'type', 'kind', 'class', 'group', 'section', 'department'
        ]
        
        # Keywords for identifying secondary detail columns (Label2)  
        self.secondary_label_keywords = [
            'quantity', 'qty', 'count', 'number', 'units', 'pieces', 'items',
            'details', 'specification', 'spec', 'notes', 'remarks', 'comment',
            'status', 'phase', 'stage', 'priority', 'urgency', 'frequency'
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
            sheets_processed = []
            
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    if df.empty:
                        continue
                    
                    # Handle empty/null/NaN values properly
                    df = df.replace([pd.NA, None, 'nan', 'NaN', 'NULL', 'null', 'None', 'N/A', 'n/a', '#N/A', '#NULL!'], '')
                    df = df.fillna('')
                    
                    # Create unique source name for each sheet
                    source_name = f"Excel_{sheet_name}"
                    
                    # Clean and process the sheet
                    processed_data = self._process_excel_sheet(df, source_name)
                    if not processed_data.empty:
                        all_budget_data.append(processed_data)
                        sheets_processed.append(sheet_name)
                        
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
                'sheets_processed': sheets_processed,
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
                content = str(cell.content).strip()
                # Handle empty/null/NaN values - keep them as empty strings
                if content.lower() in ['nan', 'null', 'none', 'n/a', '#n/a', '#null!']:
                    content = ""
                cells[cell.row_index][cell.column_index] = content
            
            # Create DataFrame
            if nrows > 1:
                headers = [self._clean_header(cell) for cell in cells[0]]
                df = pd.DataFrame(cells[1:], columns=headers)
            else:
                df = pd.DataFrame(cells)
            
            # Handle empty/null/NaN values in the DataFrame
            df = df.replace([pd.NA, None, 'nan', 'NaN', 'NULL', 'null', 'None', 'N/A', 'n/a', '#N/A', '#NULL!'], '')
            df = df.fillna('')
            
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
            
            # Handle empty/null/NaN values - keep them as empty strings, don't fill with text
            df = df.replace([pd.NA, None, 'nan', 'NaN', 'NULL', 'null', 'None'], '')
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
                        primary_label = self._extract_meaningful_label(before_amount)
                        secondary_label = self._extract_meaningful_label(after_amount)
                        
                        if primary_label or secondary_label:  # Only add if we have some context
                            currency_symbol = self._extract_currency_symbol(amount_str)
                            
                            # Determine appropriate headers based on content
                            primary_header = self._determine_text_header(before_amount, 'Description')
                            secondary_header = self._determine_text_header(after_amount, 'Details') if secondary_label else ''
                            
                            entry = {
                                primary_header: primary_label if primary_label else secondary_label,
                                'Budget_Amount': amount_value,
                                'Currency_Symbol': currency_symbol,
                                'Original_Text': amount_str,
                                'Source': source_name,
                                'Line_Number': line_idx + 1,
                                'Primary_Header': primary_header,
                                'Budget_Header': 'Amount'
                            }
                            
                            # Only add secondary column if we have meaningful content
                            if secondary_label and primary_label and secondary_header:
                                entry[secondary_header] = secondary_label
                                entry['Secondary_Header'] = secondary_header
                            
                            extracted_data.append(entry)
            
            return pd.DataFrame(extracted_data) if extracted_data else pd.DataFrame()
            
        except Exception as e:
            print(f"Error extracting from text: {e}")
            return pd.DataFrame()

    def _extract_budget_from_dataframe(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Extract budget information from a DataFrame using actual column headers."""
        try:
            if df.empty:
                return pd.DataFrame()
            
            # Convert all data to string for processing
            df = df.astype(str)
            
            # Find all potential budget columns
            budget_columns = self._find_budget_columns(df)
            
            if not budget_columns:
                return pd.DataFrame()
            
            # Identify primary and secondary label columns using actual headers
            primary_columns = self._find_primary_label_columns(df, budget_columns)
            secondary_columns = self._find_secondary_label_columns(df, budget_columns)
            
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
                    # Build primary label using actual column headers and data
                    primary_label, primary_header = self._build_primary_label(row, primary_columns)
                    
                    # Build secondary label using actual column headers and data
                    secondary_label, secondary_header = self._build_secondary_label(row, secondary_columns)
                    
                    # Create the entry with actual column headers
                    entry = {
                        primary_header: primary_label,
                        'Budget_Amount': best_budget_info['amount'],
                        'Currency_Symbol': best_budget_info['currency_symbol'],
                        'Original_Text': best_budget_info['original_text'],
                        'Source': source_name,
                        'Budget_Column': best_budget_info['column'],
                        'Primary_Header': primary_header,
                        'Budget_Header': best_budget_info['column']
                    }
                    
                    # Only add secondary column if we have meaningful content
                    if secondary_label and secondary_header:
                        entry[secondary_header] = secondary_label
                        entry['Secondary_Header'] = secondary_header
                    
                    extracted_data.append(entry)
            
            return pd.DataFrame(extracted_data) if extracted_data else pd.DataFrame()
            
        except Exception as e:
            print(f"Error extracting budget from DataFrame: {e}")
            return pd.DataFrame()

    def _find_primary_label_columns(self, df: pd.DataFrame, exclude_columns: List[str]) -> List[str]:
        """Find columns that contain primary descriptions (service, description, etc.)."""
        primary_columns = []
        available_columns = [col for col in df.columns if col not in exclude_columns]
        
        # First, look for columns with primary keywords in their names
        for col in available_columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in self.primary_label_keywords):
                primary_columns.append(col)
        
        # If no primary columns found, take the first few non-budget columns
        if not primary_columns and available_columns:
            primary_columns = available_columns[:2]  # Take first 2 columns as primary
        
        return primary_columns

    def _find_secondary_label_columns(self, df: pd.DataFrame, exclude_columns: List[str]) -> List[str]:
        """Find columns that contain secondary details (quantity, items, etc.)."""
        secondary_columns = []
        available_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Look for columns with secondary keywords in their names
        for col in available_columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in self.secondary_label_keywords):
                secondary_columns.append(col)
        
        # If no specific secondary columns found, look for columns not used as primary
        primary_columns = self._find_primary_label_columns(df, exclude_columns)
        remaining_columns = [col for col in available_columns if col not in primary_columns]
        
        if not secondary_columns and remaining_columns:
            secondary_columns = remaining_columns[:1]  # Take first remaining column
        
        return secondary_columns

    def _build_primary_label(self, row: pd.Series, primary_columns: List[str]) -> Tuple[str, str]:
        """Build primary label from service and description columns with ':' separator."""
        if not primary_columns:
            return '', 'Description'
        
        # Collect values from primary columns
        values = []
        headers = []
        
        for col in primary_columns[:2]:  # Limit to first 2 primary columns
            value = str(row[col]).strip()
            # Handle empty/null values properly - exclude them
            if value and value.lower() not in ['', 'nan', 'none', 'null', 'n/a', '#n/a', '#null!']:
                values.append(value)
                headers.append(col)
        
        if not values:
            return '', primary_columns[0] if primary_columns else 'Description'
        
        # Determine the header name - combine if multiple columns used
        if len(headers) == 1:
            header_name = headers[0]
        elif len(headers) == 2:
            header_name = f"{headers[0]} : {headers[1]}"
        else:
            header_name = "Description"
        
        # Combine values with ':' if multiple exist
        combined_value = " : ".join(values)
        
        return combined_value, header_name

    def _build_secondary_label(self, row: pd.Series, secondary_columns: List[str]) -> Tuple[str, str]:
        """Build secondary label from quantity, items, etc."""
        if not secondary_columns:
            return '', ''
        
        # Take the first meaningful secondary column
        for col in secondary_columns[:1]:  # Limit to first secondary column
            value = str(row[col]).strip()
            # Handle empty/null values properly - exclude them
            if value and value.lower() not in ['', 'nan', 'none', 'null', 'n/a', '#n/a', '#null!']:
                return value, col
        
        return '', ''

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

    def _determine_text_header(self, text: str, default: str) -> str:
        """Determine appropriate header name based on text content."""
        if not text:
            return default
        
        text_lower = text.lower()
        
        # Check for service/description keywords
        if any(keyword in text_lower for keyword in ['service', 'description', 'item', 'product']):
            return 'Service/Description'
        
        # Check for quantity/detail keywords  
        if any(keyword in text_lower for keyword in ['qty', 'quantity', 'units', 'pieces', 'count']):
            return 'Quantity'
            
        # Check for other common patterns
        if any(keyword in text_lower for keyword in ['details', 'notes', 'remarks']):
            return 'Details'
            
        return default

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

    def _ensure_max_three_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has maximum 3 columns with proper structure."""
        if df.empty:
            return df
        
        # Get non-system columns
        display_columns = [col for col in df.columns 
                         if col not in ['Currency_Symbol', 'Original_Text', 'Source', 
                                       'Budget_Column', 'Primary_Header', 'Secondary_Header', 
                                       'Budget_Header', 'Line_Number']]
        
        # If we have more than 3 display columns, prioritize
        if len(display_columns) > 3:
            # Priority: Primary label, Secondary label, Budget Amount
            budget_col = None
            primary_col = None
            secondary_col = None
            
            # Find budget column
            if 'Budget_Amount' in display_columns:
                budget_col = 'Budget_Amount'
                remaining_cols = [col for col in display_columns if col != 'Budget_Amount']
            else:
                remaining_cols = display_columns
            
            # Find primary and secondary columns
            if len(remaining_cols) >= 1:
                primary_col = remaining_cols[0]
            if len(remaining_cols) >= 2:
                secondary_col = remaining_cols[1]
            
            # Build final column list (max 3)
            final_columns = []
            if primary_col:
                final_columns.append(primary_col)
            if secondary_col:
                final_columns.append(secondary_col)
            if budget_col:
                final_columns.append(budget_col)
            
            # Keep only the selected columns
            df = df[final_columns + [col for col in df.columns if col not in display_columns]].copy()
        
        return df

    def _clean_display_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean values for display - ensure empty cells show as empty."""
        if df.empty:
            return df
        
        df_cleaned = df.copy()
        
        # Clean all non-budget columns
        for col in df_cleaned.columns:
            if col not in ['Budget_Amount', 'Currency_Symbol', 'Original_Text', 'Source', 
                          'Budget_Column', 'Primary_Header', 'Secondary_Header', 
                          'Budget_Header', 'Line_Number']:
                df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
                df_cleaned[col] = df_cleaned[col].replace(['nan', 'NaN', 'NULL', 'null', 'None', 'N/A', 'n/a', '#N/A', '#NULL!'], '')
        
        return df_cleaned

    def _combine_and_process_data(self, data_list: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple DataFrames and process to final format with dynamic column headers."""
        if not data_list:
            return pd.DataFrame(columns=['Description', 'Budget_Amount'])
        
        # Combine all DataFrames
        combined_df = pd.concat(data_list, ignore_index=True)
        
        if combined_df.empty:
            return pd.DataFrame(columns=['Description', 'Budget_Amount'])
        
        # Clean display values first
        combined_df = self._clean_display_values(combined_df)
        
        # Ensure required columns exist
        if 'Budget_Amount' not in combined_df.columns:
            combined_df['Budget_Amount'] = 0.0
        if 'Currency_Symbol' not in combined_df.columns:
            combined_df['Currency_Symbol'] = '$'
        
        # Clean and standardize
        combined_df['Budget_Amount'] = pd.to_numeric(combined_df['Budget_Amount'], errors='coerce').fillna(0)
        combined_df['Currency_Symbol'] = combined_df['Currency_Symbol'].astype(str).str.strip()
        
        # Ensure max 3 columns structure
        combined_df = self._ensure_max_three_columns(combined_df)
        
        # Determine the primary column name from the data
        primary_header = self._determine_primary_column_name(combined_df)
        secondary_header = self._determine_secondary_column_name(combined_df)
        
        # Create final DataFrame with dynamic column structure (max 3 columns)
        final_columns = [primary_header]
        
        # Add secondary column only if it contains meaningful data
        if secondary_header and self._has_meaningful_secondary_data(combined_df, secondary_header):
            final_columns.append(secondary_header)
        
        # Always add Budget_Amount and Currency_Symbol as the last columns
        final_columns.extend(['Budget_Amount', 'Currency_Symbol'])
        
        # Create result DataFrame with renamed columns
        result_df = pd.DataFrame()
        
        # Map primary column
        result_df[primary_header] = self._get_primary_column_data(combined_df)
        
        # Map secondary column if needed
        if secondary_header and secondary_header in final_columns[1:-2]:
            result_df[secondary_header] = self._get_secondary_column_data(combined_df)
        
        # Add budget columns
        result_df['Budget_Amount'] = combined_df['Budget_Amount']
        result_df['Currency_Symbol'] = combined_df['Currency_Symbol']
        
        # Remove rows with no meaningful data
        valid_rows = (
            (result_df[primary_header].astype(str).str.strip() != '') | 
            (result_df['Budget_Amount'] > 0)
        )
        
        result_df = result_df[valid_rows].copy()
        
        # Sort by budget amount (highest first)
        if not result_df.empty:
            result_df = result_df.sort_values('Budget_Amount', ascending=False).reset_index(drop=True)
        
        return result_df

    def _determine_primary_column_name(self, df: pd.DataFrame) -> str:
        """Determine the best name for the primary column based on actual data."""
        if 'Primary_Header' in df.columns:
            # Get the most common primary header
            headers = df['Primary_Header'].dropna().value_counts()
            if not headers.empty:
                return headers.index[0]
        
        # Look for common column names in the data
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['service', 'description']):
                return col
            if col_lower in ['item', 'product', 'name', 'title']:
                return col
        
        return 'Description'

    def _determine_secondary_column_name(self, df: pd.DataFrame) -> str:
        """Determine the best name for the secondary column based on actual data."""
        if 'Secondary_Header' in df.columns:
            # Get the most common secondary header
            headers = df['Secondary_Header'].dropna().value_counts()
            if not headers.empty:
                return headers.index[0]
        
        # Look for common secondary column names in the data
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['quantity', 'qty', 'count']):
                return col
            if col_lower in ['details', 'notes', 'items']:
                return col
        
        return ''

    def _has_meaningful_secondary_data(self, df: pd.DataFrame, secondary_header: str) -> bool:
        """Check if secondary column has meaningful data."""
        if secondary_header not in df.columns and 'Secondary_Header' not in df.columns:
            return False
        
        # Check if we have actual secondary data
        secondary_data = self._get_secondary_column_data(df)
        meaningful_count = sum(1 for val in secondary_data if val and str(val).strip() and str(val).lower() not in ['nan', 'none'])
        
        return meaningful_count > 0

    def _get_primary_column_data(self, df: pd.DataFrame) -> pd.Series:
        """Extract primary column data from the DataFrame."""
        # Look for the actual primary data column
        for col in df.columns:
            if col not in ['Budget_Amount', 'Currency_Symbol', 'Original_Text', 'Source', 
                          'Budget_Column', 'Primary_Header', 'Secondary_Header', 'Budget_Header', 'Line_Number']:
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in self.primary_label_keywords) or 'description' in col_lower:
                    return df[col].astype(str).str.strip()
        
        # Fallback: get first non-system column
        for col in df.columns:
            if col not in ['Budget_Amount', 'Currency_Symbol', 'Original_Text', 'Source', 
                          'Budget_Column', 'Primary_Header', 'Secondary_Header', 'Budget_Header', 'Line_Number']:
                return df[col].astype(str).str.strip()
        
        return pd.Series([''] * len(df))

    def _get_secondary_column_data(self, df: pd.DataFrame) -> pd.Series:
        """Extract secondary column data from the DataFrame."""
        # Look for the actual secondary data column
        used_primary = None
        for col in df.columns:
            if col not in ['Budget_Amount', 'Currency_Symbol', 'Original_Text', 'Source', 
                          'Budget_Column', 'Primary_Header', 'Secondary_Header', 'Budget_Header', 'Line_Number']:
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in self.primary_label_keywords) or 'description' in col_lower:
                    used_primary = col
                    break
        
        # Find secondary column (different from primary)
        for col in df.columns:
            if (col != used_primary and 
                col not in ['Budget_Amount', 'Currency_Symbol', 'Original_Text', 'Source', 
                           'Budget_Column', 'Primary_Header', 'Secondary_Header', 'Budget_Header', 'Line_Number']):
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in self.secondary_label_keywords):
                    return df[col].astype(str).str.strip()
        
        # Fallback: get second non-system column
        non_system_cols = [col for col in df.columns if col not in 
                          ['Budget_Amount', 'Currency_Symbol', 'Original_Text', 'Source', 
                           'Budget_Column', 'Primary_Header', 'Secondary_Header', 'Budget_Header', 'Line_Number']]
        
        if len(non_system_cols) > 1 and used_primary:
            remaining_cols = [col for col in non_system_cols if col != used_primary]
            if remaining_cols:
                return df[remaining_cols[0]].astype(str).str.strip()
        
        return pd.Series([''] * len(df))

    def _find_highest_budget_row(self, df: pd.DataFrame) -> Optional[Dict]:
        """Find the row with the highest budget amount."""
        if df.empty or 'Budget_Amount' not in df.columns:
            return None
        
        max_idx = df['Budget_Amount'].idxmax()
        max_row = df.loc[max_idx]
        
        # Get actual column names dynamically
        primary_col = None
        secondary_col = None
        
        # Find the primary and secondary columns by checking column headers
        for col in df.columns:
            if col not in ['Budget_Amount', 'Currency_Symbol', 'Original_Text', 'Source', 
                          'Budget_Column', 'Primary_Header', 'Secondary_Header', 
                          'Budget_Header', 'Line_Number']:
                if primary_col is None:
                    primary_col = col
                elif secondary_col is None:
                    secondary_col = col
                    break
        
        result = {
            'index': max_idx,
            'budget_amount': max_row['Budget_Amount'],
            'row_data': max_row.to_dict()
        }
        
        # Add primary label (always present)
        if primary_col and primary_col in max_row:
            result['label1'] = max_row[primary_col]
            result['primary_header'] = primary_col
        
        # Only add secondary label if it exists and has content
        if secondary_col and secondary_col in max_row and pd.notna(max_row.get(secondary_col)) and max_row.get(secondary_col):
            result['label2'] = max_row[secondary_col]
            result['secondary_header'] = secondary_col
        
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
            combined_data = pd.DataFrame(columns=['Budget_Amount', 'Source_File', 'Currency_Symbol'])
        
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
    Budget_Amount is always the last column, secondary label only if present.
    Columns are named with actual headers from the source files.
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
                # Check if we have multiple tables/sheets to display
                sources = extracted_data['Source'].unique() if 'Source' in extracted_data.columns else ['Default']
                
                # Handle multiple tables from PDF/Image or multiple Excel sheets
                if len(sources) > 1:
                    st.subheader("📊 Extracted Budget Data")
                    st.write(f"**Found {len(sources)} tables/sheets with budget data**")
                    
                    # Display tables in 3-column layout
                    tables_per_row = 3
                    table_groups = []
                    current_group = []
                    
                    for i, source in enumerate(sources):
                        source_data = extracted_data[extracted_data['Source'] == source]
                        current_group.append((source, source_data))
                        
                        if len(current_group) == tables_per_row or i == len(sources) - 1:
                            table_groups.append(current_group)
                            current_group = []
                    
                    # Display each group of tables in columns
                    for group_idx, table_group in enumerate(table_groups):
                        cols = st.columns(len(table_group))
                        
                        for col_idx, (source, source_data) in enumerate(table_group):
                            with cols[col_idx]:
                                # Clean source name for display
                                if source.startswith('PDF_Table_'):
                                    display_name = f"Table {source.split('_')[-1]}"
                                elif source.startswith('Excel_'):
                                    sheet_name = source.replace('Excel_', '')
                                    display_name = f"Sheet: {sheet_name}"
                                else:
                                    display_name = source
                                
                                st.markdown(f"**{display_name}**")
                                
                                # Prepare display data with proper empty value handling
                                display_columns = [col for col in source_data.columns 
                                                 if col not in ['Currency_Symbol', 'Original_Text', 'Source', 
                                                               'Budget_Column', 'Primary_Header', 'Secondary_Header', 
                                                               'Budget_Header', 'Line_Number']]
                                
                                display_df = source_data[display_columns].copy()
                                
                                # Clean empty values - ensure they show as empty cells
                                for col in display_df.columns:
                                    if col != 'Budget_Amount':
                                        display_df[col] = display_df[col].astype(str).str.strip()
                                        display_df[col] = display_df[col].replace(['nan', 'NaN', 'NULL', 'null', 'None', 'N/A', 'n/a'], '')
                                
                                # Format the budget column with currency symbol
                                if 'Budget_Amount' in display_df.columns and 'Currency_Symbol' in source_data.columns:
                                    display_df['Budget_Amount'] = source_data.apply(
                                        lambda row: f"{row['Currency_Symbol']}{row['Budget_Amount']:,.2f}" if row['Budget_Amount'] > 0 else '', axis=1
                                    )
                                
                                # Ensure we have max 3 columns and Budget_Amount is last
                                if len(display_df.columns) > 3:
                                    non_budget_cols = [col for col in display_df.columns if col != 'Budget_Amount'][:2]
                                    if 'Budget_Amount' in display_df.columns:
                                        final_cols = non_budget_cols + ['Budget_Amount']
                                    else:
                                        final_cols = non_budget_cols[:3]
                                    display_df = display_df[final_cols]
                                elif 'Budget_Amount' in display_df.columns:
                                    budget_col = display_df.pop('Budget_Amount')
                                    display_df['Budget_Amount'] = budget_col
                                
                                st.dataframe(display_df, use_container_width=True, height=200)
                                
                                # Download button for individual table
                                csv_data = source_data.to_csv(index=False)
                                st.download_button(
                                    label=f"� CSV",
                                    data=csv_data,
                                    file_name=f"budget_data_{display_name.replace(' ', '_').replace(':', '')}.csv",
                                    mime="text/csv",
                                    key=f"download_{source}"
                                )
                else:
                    # Single table display
                    st.subheader("�📊 Extracted Budget Data")
                    st.write(f"**Total entries found:** {len(extracted_data)}")
                    
                    # Display with dynamic columns based on actual headers
                    display_columns = [col for col in extracted_data.columns 
                                     if col not in ['Currency_Symbol', 'Original_Text', 'Source', 
                                                   'Budget_Column', 'Primary_Header', 'Secondary_Header', 
                                                   'Budget_Header', 'Line_Number']]
                    
                    # Create display DataFrame with formatted amounts
                    display_df = extracted_data[display_columns].copy()
                    
                    # Clean empty values - ensure they show as empty cells
                    for col in display_df.columns:
                        if col != 'Budget_Amount':
                            display_df[col] = display_df[col].astype(str).str.strip()
                            display_df[col] = display_df[col].replace(['nan', 'NaN', 'NULL', 'null', 'None', 'N/A', 'n/a'], '')
                    
                    # Format the budget column with currency symbol
                    if 'Budget_Amount' in display_df.columns and 'Currency_Symbol' in extracted_data.columns:
                        display_df['Budget_Amount'] = extracted_data.apply(
                            lambda row: f"{row['Currency_Symbol']}{row['Budget_Amount']:,.2f}" if row['Budget_Amount'] > 0 else '', axis=1
                        )
                    
                    # Ensure max 3 columns with Budget_Amount last
                    if len(display_df.columns) > 3:
                        non_budget_cols = [col for col in display_df.columns if col != 'Budget_Amount'][:2]
                        if 'Budget_Amount' in display_df.columns:
                            final_cols = non_budget_cols + ['Budget_Amount']
                        else:
                            final_cols = non_budget_cols[:3]
                        display_df = display_df[final_cols]
                    elif 'Budget_Amount' in display_df.columns:
                        # Reorder columns to put Budget_Amount last
                        budget_col = display_df.pop('Budget_Amount')
                        display_df['Budget_Amount'] = budget_col
                    
                    st.dataframe(display_df, use_container_width=True)
                
                # Show column information
                if st.expander("📋 Column Information"):
                    col_info = []
                    sample_row = extracted_data.iloc[0] if not extracted_data.empty else {}
                    display_columns = [col for col in extracted_data.columns 
                                     if col not in ['Currency_Symbol', 'Original_Text', 'Source', 
                                                   'Budget_Column', 'Primary_Header', 'Secondary_Header', 
                                                   'Budget_Header', 'Line_Number']]
                    
                    for col in display_columns[:3]:  # Show info for max 3 columns
                        if col != 'Budget_Amount':
                            col_info.append(f"**{col}:** Data from original file columns")
                        else:
                            col_info.append(f"**{col}:** Highest amount selected from available budget columns")
                    st.write("\n".join(col_info))
                
                # Highlight highest budget
                highest_budget = result.get('highest_budget_row')
                if highest_budget:
                    st.subheader("🏆 Highest Budget Entry")
                    currency_symbol = extracted_data.loc[highest_budget['index'], 'Currency_Symbol'] if 'Currency_Symbol' in extracted_data.columns else '$'
                    
                    # Get the actual column names from the data
                    display_columns = [col for col in extracted_data.columns 
                                     if col not in ['Currency_Symbol', 'Original_Text', 'Source', 
                                                   'Budget_Column', 'Primary_Header', 'Secondary_Header', 
                                                   'Budget_Header', 'Line_Number', 'Budget_Amount']]
                    
                    info_text = ""
                    row_data = extracted_data.loc[highest_budget['index']]
                    
                    # Display each meaningful column (max 3 columns)
                    for col in display_columns[:2]:  # Show max 2 non-budget columns
                        if pd.notna(row_data[col]) and str(row_data[col]).strip() and str(row_data[col]).strip() not in ['nan', 'NaN', 'NULL', 'null', 'None', 'N/A', 'n/a']:
                            info_text += f"**{col}:** {row_data[col]}\n\n"
                    
                    info_text += f"**Amount:** {currency_symbol}{highest_budget['budget_amount']:,.2f}"
                    
                    st.info(info_text)
                
                # Show additional metadata
                if st.expander("📋 Processing Details"):
                    st.write(f"**File Type:** {result.get('file_type', 'Unknown')}")
                    st.write(f"**Budget Columns Found:** {result.get('budget_columns_found', [])}")
                    if 'sheets_processed' in result:
                        st.write(f"**Excel Sheets Processed:** {result['sheets_processed']}")
                
                # Download option for all data
                csv_data = extracted_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download All Results as CSV",
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
            
            # Create display DataFrame with formatted amounts - exclude system columns
            display_columns = [col for col in top_entries.columns 
                             if col not in ['Currency_Symbol', 'Original_Text', 'Source', 
                                           'Budget_Column', 'Primary_Header', 'Secondary_Header', 
                                           'Budget_Header', 'Line_Number']]
            
            display_df = top_entries[display_columns].copy()
            
            # Format the budget column with currency symbol
            if 'Budget_Amount' in display_df.columns and 'Currency_Symbol' in top_entries.columns:
                display_df['Budget_Amount'] = top_entries.apply(
                    lambda row: f"{row['Currency_Symbol']}{row['Budget_Amount']:,.2f}", axis=1
                )
            
            # Ensure Budget_Amount is last column
            if 'Budget_Amount' in display_df.columns:
                budget_col = display_df.pop('Budget_Amount')
                display_df['Budget_Amount'] = budget_col
            
            st.dataframe(display_df, use_container_width=True)
            
            # Overall highest budget
            highest_overall = result.get('highest_budget_overall')
            if highest_overall:
                st.subheader("🏆 Overall Highest Budget Entry")
                source_file = combined_data.loc[highest_overall['index'], 'Source_File']
                currency_symbol = combined_data.loc[highest_overall['index'], 'Currency_Symbol'] if 'Currency_Symbol' in combined_data.columns else '$'
                
                # Get dynamic header names
                primary_header = highest_overall.get('primary_header', 'Primary Label')
                info_text = f"**{primary_header}:** {highest_overall['label1']}\n\n"
                
                # Only show secondary label if it exists and has content
                if 'label2' in highest_overall and highest_overall['label2']:
                    secondary_header = highest_overall.get('secondary_header', 'Secondary Label')
                    info_text += f"**{secondary_header}:** {highest_overall['label2']}\n\n"
                
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
