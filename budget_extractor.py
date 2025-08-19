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
from utils.currency_utils import MONEY_PATTERN, MONEY_KEYWORDS, contains_money
from ui_handler import standardize_dataframe, preprocess_dataframe, add_download_buttons, deduplicate_columns, download_table_as_json
import os
from dotenv import load_dotenv

class BudgetExtractor:
    """Budget and label extraction from PDFs, images, and Excel files."""
    
    def __init__(self):
        load_dotenv()
        self.config = AzureConfig()
        self.document_client = self.config.get_document_client()
        self.blob_service_client = self.config.get_blob_service_client()
        self.blob_manager = BlobManager(self.blob_service_client, self.config.blob_container)
        self.table_extractor = TableExtractor(self.document_client)
        
        # Common patterns and keywords
        self.currency_symbols = ['$', '€', '£', '¥', '₹', '₩', '₽', 'Rs', 'USD', 'EUR', 'GBP', 'INR']
        self.budget_keywords = [
            'budget', 'amount', 'cost', 'price', 'total', 'value', 'expense', 
            'revenue', 'income', 'salary', 'wage', 'fee', 'payment', 'sum',
            'balance', 'fund', 'allocation', 'expenditure', 'spend', 'investment'
        ]
        self.primary_label_keywords = [
            'service', 'description', 'item', 'product', 'name', 'title', 
            'project', 'task', 'activity', 'work', 'job', 'category',
            'type', 'kind', 'class', 'group', 'section', 'department'
        ]
        self.secondary_label_keywords = [
            'quantity', 'qty', 'count', 'number', 'units', 'pieces', 'items',
            'details', 'specification', 'spec', 'notes', 'remarks', 'comment',
            'status', 'phase', 'stage', 'priority', 'urgency', 'frequency'
        ]
        
        # Address detection patterns to exclude from amounts
        self.address_patterns = [
            # Street addresses (house numbers + street names)
            r"\b\d+\s+(main|first|second|third|oak|elm|park|church|school|high|broad|union|washington|lincoln|jefferson|madison|jackson|adams|monroe|harrison|tyler|polk|taylor|fillmore|pierce|buchanan|johnson|grant|hayes|garfield|cleveland|harrison|mckinley|roosevelt|taft|wilson|harding|coolidge|hoover|truman|eisenhower|kennedy|nixon|ford|carter|reagan|bush|clinton|obama|trump|biden)\s+(st|street|ave|avenue|rd|road|blvd|boulevard|ln|lane|dr|drive|ct|court|pl|place|way|circle|square|plaza|terrace|trail|path|walk|row|crescent|grove|heights|hill|park|view)\b",
            # Common street suffixes with numbers
            r"\b\d+\s+\w+\s+(st|street|ave|avenue|rd|road|blvd|boulevard|ln|lane|dr|drive|ct|court|pl|place|way|circle|square|plaza|terrace|trail|path|walk|row|crescent|grove|heights|hill|park|view)\b",
            # ZIP codes
            r"\b\d{5}(-\d{4})?\b",
            # PO Box
            r"\bP\.?O\.?\s*Box\s*\d+\b",
            # Suite/Apartment numbers
            r"\b(suite|apt|apartment|unit|floor|room)\s*#?\s*\d+\b",
            # Building numbers
            r"\bbuilding\s+\d+\b",
            # House numbers (isolated numbers that are likely addresses)
            r"\b\d{1,5}\s+[A-Z][a-z]+\s+(St|Ave|Rd|Dr|Ln|Ct|Pl|Way|Blvd)\b",
        ]

        # Combined address pattern
        self.combined_address_pattern = re.compile("|".join(self.address_patterns), re.IGNORECASE)

        # Null/empty value patterns for cleaning
        self.null_patterns = [pd.NA, None, "nan", "NaN", "NULL", "null", "None", "N/A", "n/a", "#N/A", "#NULL!"]

    def extract_budget_data(self, file_path: str = None, file_bytes: bytes = None, file_extension: str = None) -> Dict[str, Any]:
        """Extract budget data from various file formats."""
        try:
            if file_path:
                file_bytes = self.blob_manager.download_file(file_path)
                file_extension = os.path.splitext(file_path)[1].lower()
            
            if not file_bytes or not file_extension:
                raise ValueError("Either file_path or both file_bytes and file_extension must be provided")
            
            # Route to appropriate extraction method
            extraction_methods = {
                '.pdf': self._extract_from_pdf,
                '.xlsx': self._extract_from_excel,
                '.xls': self._extract_from_excel,
                '.jpg': self._extract_from_image,
                '.jpeg': self._extract_from_image,
                '.png': self._extract_from_image,
                '.tiff': self._extract_from_image,
                '.bmp': self._extract_from_image
            }
            
            method = extraction_methods.get(file_extension)
            if method:
                return method(file_bytes, file_path or f"Document{file_extension}")
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            return self._create_error_result(str(e), file_path or "Unknown")

    def _create_error_result(self, error: str, file_name: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'success': False,
            'error': error,
            'file_name': file_name,
            'extracted_data': pd.DataFrame()
        }

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Common DataFrame cleaning operations."""
        df = df.replace(self.null_patterns, '')
        df = df.fillna('')
        return df

    def _extract_from_pdf(self, pdf_bytes: bytes, file_name: str) -> Dict[str, Any]:
        """Extract budget data from PDF files."""
        try:
            # Delegate table extraction to TableExtractor (avoids duplicating parsing logic)
            tables = self.table_extractor.extract_from_pdf(pdf_bytes if isinstance(pdf_bytes, (bytes, bytearray)) else pdf_bytes.read())
            all_budget_data = []

            for i, table_df in enumerate(tables):
                # Use the shared cleaner from TableExtractor and local dataframe cleaning
                try:
                    table_df = TableExtractor.clean_table(table_df)
                except Exception:
                    pass

                table_df = self._clean_dataframe(table_df)
                processed = self._extract_budget_from_dataframe(table_df, f"PDF_Table_{i}")
                if not processed.empty:
                    all_budget_data.append(processed)

            # Note: text-based extraction (inline amounts outside tables) is intentionally omitted here
            # to avoid duplicating Azure Document Intelligence calls; keep focus on table extraction and
            # reuse shared table utilities from table_utils.

            return self._create_success_result(all_budget_data, file_name, 'PDF')
            
        except Exception as e:
            return self._create_error_result(str(e), file_name)

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
                    
                    df = self._clean_dataframe(df)
                    processed_data = self._process_excel_sheet(df, f"Excel_{sheet_name}")
                    if not processed_data.empty:
                        all_budget_data.append(processed_data)
                        sheets_processed.append(sheet_name)
                        
                except Exception as sheet_error:
                    print(f"Error processing sheet {sheet_name}: {sheet_error}")
                    continue
            
            result = self._create_success_result(all_budget_data, file_name, 'Excel')
            result['sheets_processed'] = sheets_processed
            return result
            
        except Exception as e:
            return self._create_error_result(str(e), file_name)

    def _extract_from_image(self, image_bytes: bytes, file_name: str) -> Dict[str, Any]:
        """Extract budget data from image files."""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to PDF for processing
            pdf_bytes = io.BytesIO()
            image.save(pdf_bytes, format="PDF", quality=95)
            pdf_bytes.seek(0)
            
            return self._extract_from_pdf(pdf_bytes.read(), file_name)
            
        except Exception as e:
            return self._create_error_result(str(e), file_name)

    def _create_success_result(self, all_budget_data: List[pd.DataFrame], file_name: str, file_type: str) -> Dict[str, Any]:
        """Create standardized success result."""
        final_data = self._combine_and_process_data(all_budget_data)
        return {
            'success': True,
            'file_name': file_name,
            'file_type': file_type,
            'extracted_data': final_data,
            'highest_budget_row': self._find_highest_budget_row(final_data),
            'total_rows': len(final_data),
            'budget_columns_found': self._get_budget_columns(final_data)
        }

    def _process_table_structure(self, table, source_name: str) -> pd.DataFrame:
        """Process Azure Document Intelligence table structure.

        Deprecated: table parsing is handled by TableExtractor.extract_from_pdf now.
        This stub remains for backward compatibility.
        """
        # Deprecated: table parsing now handled by TableExtractor.extract_from_pdf
        return pd.DataFrame()

    def _process_excel_sheet(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Process Excel sheet to extract budget information."""
        try:
            df = df.dropna(how='all').reset_index(drop=True)
            df = df.loc[:, ~df.columns.duplicated()]
            df = self._clean_dataframe(df)
            return self._extract_budget_from_dataframe(df, source_name)
            
        except Exception as e:
            print(f"Error processing Excel sheet: {e}")
            return pd.DataFrame()

    def _extract_from_text(self, text: str, source_name: str) -> pd.DataFrame:
        """Extract budget information from plain text."""
        try:
            extracted_data = []
            
            for line_idx, line in enumerate(text.split('\n')):
                line = line.strip()
                if not line:
                    continue
                
                # Skip lines that appear to contain addresses
                if self._is_likely_address(line):
                    continue
                
                money_matches = list(MONEY_PATTERN.finditer(line))
                
                if money_matches:
                    for match in money_matches:
                        amount_str = match.group()
                        amount_value = self._parse_amount_value(amount_str)
                        
                        before_amount = line[:match.start()].strip()
                        after_amount = line[match.end():].strip()
                        
                        # Additional check: skip if the context around the amount suggests it's an address
                        full_context = f"{before_amount} {amount_str} {after_amount}"
                        if self._is_likely_address(full_context):
                            continue
                        
                        primary_label = self._extract_meaningful_label(before_amount)
                        secondary_label = self._extract_meaningful_label(after_amount)
                        
                        if primary_label or secondary_label:
                            currency_symbol = self._extract_currency_symbol(amount_str)
                            primary_header = self._determine_text_header(before_amount, 'Description')
                            secondary_header = self._determine_text_header(after_amount, 'Details') if secondary_label else ''
                            
                            entry = {
                                primary_header: primary_label if primary_label else secondary_label,
                                'Budget_Amount': amount_value,
                                'Currency_Symbol': currency_symbol,
                                'Original_Text': line,
                                'Source': source_name,
                                'Line_Number': line_idx + 1,
                                'Primary_Header': primary_header,
                                'Budget_Header': 'Amount'
                            }
                            
                            if secondary_label and primary_label and secondary_header:
                                entry[secondary_header] = secondary_label
                                entry['Secondary_Header'] = secondary_header
                            
                            extracted_data.append(entry)
            
            return pd.DataFrame(extracted_data) if extracted_data else pd.DataFrame()
            
        except Exception as e:
            print(f"Error extracting from text: {e}")
            return pd.DataFrame()

    def _extract_budget_from_dataframe(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Extract budget information from DataFrame."""
        if df.empty:
            return pd.DataFrame()

        try:
            # Find budget columns
            budget_columns = self._find_budget_columns(df)
            if not budget_columns:
                return pd.DataFrame()

            # Find label columns
            primary_columns = self._find_primary_label_columns(df, budget_columns)
            secondary_columns = self._find_secondary_label_columns(df, budget_columns + primary_columns)

            extracted_data = []
            best_budget_info = {'column': budget_columns[0], 'score': 0}

            for _, row in df.iterrows():
                # Get budget amount
                budget_amount = 0
                for budget_col in budget_columns:
                    raw_value = str(row[budget_col]).strip() if pd.notna(row[budget_col]) else ''
                    if self._is_numeric_amount(raw_value):
                        budget_amount = self._parse_amount_value(raw_value)
                        best_budget_info = {'column': budget_col, 'score': budget_amount}
                        break

                if budget_amount <= 0:
                    continue

                # Extract labels
                primary_label, primary_header = self._build_primary_label(row, primary_columns)
                secondary_label, secondary_header = self._build_secondary_label(row, secondary_columns)
                
                # Skip entries where labels appear to be addresses
                if (primary_label and self._is_likely_address(primary_label)) or \
                   (secondary_label and self._is_likely_address(secondary_label)):
                    continue

                currency_symbol = self._extract_currency_symbol(str(row[best_budget_info['column']]))

                entry = {
                    primary_header: primary_label,
                    'Budget_Amount': budget_amount,
                    'Currency_Symbol': currency_symbol,
                    'Source': source_name,
                    'Primary_Header': primary_header,
                    'Budget_Header': best_budget_info['column']
                }

                if secondary_label and secondary_header:
                    entry[secondary_header] = secondary_label
                    entry['Secondary_Header'] = secondary_header

                extracted_data.append(entry)

            return pd.DataFrame(extracted_data) if extracted_data else pd.DataFrame()

        except Exception as e:
            print(f"Error extracting budget from dataframe: {e}")
            return pd.DataFrame()

    def _find_budget_columns(self, df: pd.DataFrame) -> List[str]:
        """Find columns containing budget/amount data."""
        budget_columns = []
        
        for col in df.columns:
            col_str = str(col).lower()
            # Check column name
            if any(keyword in col_str for keyword in self.budget_keywords):
                budget_columns.append(col)
                continue
            
            # Check column values
            numeric_count = 0
            for value in df[col].dropna().head(10):
                if self._is_numeric_amount(str(value)):
                    numeric_count += 1
            
            if numeric_count >= 2:
                budget_columns.append(col)
        
        return budget_columns

    def _find_primary_label_columns(self, df: pd.DataFrame, exclude_columns: List[str]) -> List[str]:
        """Find primary label columns."""
        primary_columns = []
        
        for col in df.columns:
            if col in exclude_columns:
                continue
            
            col_str = str(col).lower()
            if any(keyword in col_str for keyword in self.primary_label_keywords):
                primary_columns.append(col)
            elif not col_str.replace('_', '').replace('-', '').replace(' ', '').isdigit():
                # Include non-numeric column names
                primary_columns.append(col)
        
        return primary_columns[:3]  # Limit to 3 columns

    def _find_secondary_label_columns(self, df: pd.DataFrame, exclude_columns: List[str]) -> List[str]:
        """Find secondary label columns."""
        secondary_columns = []
        
        for col in df.columns:
            if col in exclude_columns:
                continue
            
            col_str = str(col).lower()
            if any(keyword in col_str for keyword in self.secondary_label_keywords):
                secondary_columns.append(col)
        
        return secondary_columns[:2]  # Limit to 2 columns

    def _build_primary_label(self, row: pd.Series, primary_columns: List[str]) -> Tuple[str, str]:
        """Build primary label from row data."""
        for col in primary_columns:
            value = str(row[col]).strip() if pd.notna(row[col]) else ''
            if value and value not in ['nan', 'NaN', 'NULL', 'null', 'None', 'N/A', 'n/a']:
                return value, col
        
        return 'Item', primary_columns[0] if primary_columns else 'Description'

    def _build_secondary_label(self, row: pd.Series, secondary_columns: List[str]) -> Tuple[str, str]:
        """Build secondary label from row data."""
        for col in secondary_columns:
            value = str(row[col]).strip() if pd.notna(row[col]) else ''
            if value and value not in ['nan', 'NaN', 'NULL', 'null', 'None', 'N/A', 'n/a']:
                return value, col
        
        return '', ''

    def _is_numeric_amount(self, value: str) -> bool:
        """Check if value represents a numeric amount (not just any number)."""
        if not value or str(value).strip() == '':
            return False
        
        value_str = str(value).strip()
        
        # Skip if it looks like an address component
        if self._is_likely_address(value_str):
            return False
            
        # Must have currency indicator or be in a clearly monetary context
        has_currency_symbol = any(symbol in value_str for symbol in self.currency_symbols)
        has_decimal = '.' in value_str and len(value_str.split('.')[-1]) <= 2
        
        # Clean the value
        clean_value = re.sub(r'[^\d.,]', '', value_str)
        if not clean_value:
            return False
        
        try:
            numeric_value = float(clean_value.replace(',', ''))
            
            # Additional checks for valid monetary amounts
            # Skip very small numbers unless they have clear currency context
            if numeric_value < 1.0 and not has_currency_symbol:
                return False
                
            # Skip very large numbers that are likely not money (unless clearly marked)
            if numeric_value > 999999999 and not has_currency_symbol:
                return False
                
            return has_currency_symbol or has_decimal or numeric_value >= 1.0
            
        except ValueError:
            return False

    def _extract_currency_symbol(self, amount_str: str) -> str:
        """Extract currency symbol from amount string."""
        amount_str = str(amount_str).strip()
        
        # Check for explicit symbols
        for symbol in self.currency_symbols:
            if symbol in amount_str:
                return symbol if len(symbol) <= 3 else '$'
        
        return '$'  # Default to USD

    def _parse_amount_value(self, amount_str: str) -> float:
        """Parse amount string to float value."""
        try:
            amount_str = str(amount_str).strip()
            
            # Handle multipliers
            multipliers = {'k': 1000, 'm': 1000000, 'b': 1000000000}
            multiplier = 1
            
            for suffix, mult in multipliers.items():
                if amount_str.lower().endswith(suffix):
                    multiplier = mult
                    amount_str = amount_str[:-1]
                    break
            
            # Extract numeric part
            numeric_part = re.sub(r'[^\d.,]', '', amount_str)
            if not numeric_part:
                return 0.0
            
            # Handle comma as thousands separator
            if ',' in numeric_part and '.' in numeric_part:
                parts = numeric_part.split('.')
                if len(parts) == 2 and len(parts[1]) <= 2:
                    numeric_part = parts[0].replace(',', '') + '.' + parts[1]
                else:
                    numeric_part = numeric_part.replace(',', '')
            elif ',' in numeric_part:
                numeric_part = numeric_part.replace(',', '')
            
            return float(numeric_part) * multiplier
            
        except (ValueError, AttributeError):
            return 0.0

    def _determine_text_header(self, text: str, default: str) -> str:
        """Determine appropriate header name from text context."""
        if not text:
            return default
        
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ['service', 'task', 'work']):
            return 'Service'
        elif any(keyword in text_lower for keyword in ['item', 'product', 'goods']):
            return 'Item'
        elif any(keyword in text_lower for keyword in ['description', 'desc']):
            return 'Description'
        elif any(keyword in text_lower for keyword in ['category', 'type', 'class']):
            return 'Category'
        elif any(keyword in text_lower for keyword in ['details', 'notes', 'remarks']):
            return 'Details'
            
        return default

    def _is_likely_address(self, text: str) -> bool:
        """Check if text contains address patterns that should be excluded from amount extraction."""
        if not text:
            return False
        
        # Check for common address patterns
        return bool(self.combined_address_pattern.search(text))

    def _extract_meaningful_label(self, text: str) -> str:
        """Extract meaningful label from text context."""
        if not text:
            return ''
        
        # Remove common noise words
        noise_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word.strip() for word in text.split() if word.strip().lower() not in noise_words]
        
        meaningful_words = []
        for word in words:
            if len(word) > 1 and not word.isdigit():
                meaningful_words.append(word)
            if len(meaningful_words) >= 3:
                break
        
        return ' '.join(meaningful_words)

    def _clean_header(self, header: str) -> str:
        """Clean column header."""
        if not header:
            return 'Unknown'
        return str(header).strip().replace('\n', ' ').replace('\r', '')

    def _combine_and_process_data(self, data_list: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple DataFrames and process to final format."""
        if not data_list:
            return pd.DataFrame()
        
        combined_df = pd.concat(data_list, ignore_index=True)
        if combined_df.empty:
            return pd.DataFrame()
        
        # Clean and standardize
        if 'Budget_Amount' not in combined_df.columns:
            combined_df['Budget_Amount'] = 0.0
        if 'Currency_Symbol' not in combined_df.columns:
            combined_df['Currency_Symbol'] = '$'
        
        combined_df['Budget_Amount'] = pd.to_numeric(combined_df['Budget_Amount'], errors='coerce').fillna(0)
        combined_df = combined_df[combined_df['Budget_Amount'] > 0].copy()
        
        if not combined_df.empty:
            combined_df = combined_df.sort_values('Budget_Amount', ascending=False).reset_index(drop=True)
        
        return combined_df

    def _find_highest_budget_row(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Find the row with highest budget amount."""
        if df.empty or 'Budget_Amount' not in df.columns:
            return None
        
        max_idx = df['Budget_Amount'].idxmax()
        max_row = df.loc[max_idx]
        
        return {
            'index': max_idx,
            'budget_amount': max_row['Budget_Amount'],
            'currency_symbol': max_row.get('Currency_Symbol', '$'),
            'row_data': max_row.to_dict()
        }

    def _get_budget_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of budget-related columns."""
        budget_cols = []
        for col in df.columns:
            if 'budget' in str(col).lower() or 'amount' in str(col).lower():
                budget_cols.append(col)
        return budget_cols

    def _get_dominant_currency(self, df: pd.DataFrame) -> str:
        """Get the dominant currency symbol, prioritizing USD."""
        if df.empty or 'Currency_Symbol' not in df.columns:
            return '$'
        
        currency_counts = df['Currency_Symbol'].value_counts()
        
        # Prioritize USD ($)
        if '$' in currency_counts.index:
            return '$'
        elif 'USD' in currency_counts.index:
            return '$'
        
        # Return most frequent currency
        return currency_counts.index[0] if not currency_counts.empty else '$'

    def _standardize_currency_display(self, amount: float, currency: str = None) -> str:
        """Format amount with standardized currency display."""
        if pd.isna(amount) or amount <= 0:
            return ''
        
        currency = currency or '$'
        return f"{currency}{amount:,.2f}"

    def _compare_table_labels(self, tables_data: Dict[str, pd.DataFrame]) -> Dict[str, List[Tuple[str, pd.DataFrame]]]:
        """Compare table labels and group similar tables - only exact header matches."""
        if not tables_data:
            return {}
        
        label_groups = {}
        
        for source, data in tables_data.items():
            if data.empty:
                continue
            
            # Get actual column headers from the data
            display_columns = [col for col in data.columns 
                             if col not in ['Currency_Symbol', 'Original_Text', 'Source', 
                                           'Budget_Column', 'Primary_Header', 'Secondary_Header', 
                                           'Budget_Header', 'Line_Number']]
            
            # Create signature based on exact column names
            column_signature = tuple(sorted(display_columns))
            
            if column_signature not in label_groups:
                label_groups[column_signature] = []
            
            label_groups[column_signature].append((source, data))
        
        return label_groups

    def _merge_compatible_tables(self, table_group: List[Tuple[str, pd.DataFrame]]) -> List[Tuple[str, pd.DataFrame]]:
        """Merge tables with exactly matching headers."""
        if len(table_group) <= 1:
            return table_group
        
        # Check if all tables have exactly the same structure
        first_table = table_group[0][1]
        first_columns = set(col for col in first_table.columns 
                           if col not in ['Currency_Symbol', 'Original_Text', 'Source', 
                                         'Budget_Column', 'Primary_Header', 'Secondary_Header', 
                                         'Budget_Header', 'Line_Number'])
        
        # Only merge if ALL tables have identical column structure
        compatible_tables = []
        for source, data in table_group:
            data_columns = set(col for col in data.columns 
                             if col not in ['Currency_Symbol', 'Original_Text', 'Source', 
                                           'Budget_Column', 'Primary_Header', 'Secondary_Header', 
                                           'Budget_Header', 'Line_Number'])
            
            if data_columns == first_columns:
                compatible_tables.append((source, data))
            else:
                # Return unmerged if any table has different structure
                return table_group
        
        # Merge compatible tables
        if len(compatible_tables) > 1:
            merged_data = pd.concat([data for _, data in compatible_tables], ignore_index=True)
            source_names = [source for source, _ in compatible_tables]
            merged_source_name = f"Merged: {', '.join(source_names)}"
            return [(merged_source_name, merged_data)]
        
        return table_group


def display_budget_results(result, single_file=True):
    """Display budget extraction results in Streamlit with 3-column layout."""
    if single_file:
        if result.get('success', False):
            extracted_data = result.get('extracted_data')
            if not extracted_data.empty:
                # Standardize currency display
                extractor = BudgetExtractor()
                dominant_currency = extractor._get_dominant_currency(extracted_data)
                
                # Get sources and prepare for display
                sources = extracted_data['Source'].unique() if 'Source' in extracted_data.columns else ['Default']
                
                st.subheader("Extracted Budget Data")
                
                # Prepare tables data for comparison
                tables_data = {}
                for source in sources:
                    tables_data[source] = extracted_data[extracted_data['Source'] == source]
                
                # Compare labels and group similar tables
                label_groups = extractor._compare_table_labels(tables_data)

                # Process each label group
                # NOTE: do NOT merge tables here; keep each extracted table separate
                final_tables = []
                for label_signature, table_group in label_groups.items():
                    for source, data in table_group:
                        # append a shallow copy to avoid accidental shared-state edits
                        final_tables.append((source, data.copy()))
                
                # Display tables in 3-column layout
                tables_per_row = 3
                table_groups = []
                current_group = []
                
                for i, (source, source_data) in enumerate(final_tables):
                    current_group.append((source, source_data))
                    
                    if len(current_group) == tables_per_row or i == len(final_tables) - 1:
                        table_groups.append(current_group)
                        current_group = []
                
                # Display each group of tables in columns
                for table_group in table_groups:
                    cols = st.columns(len(table_group))
                    
                    for col_idx, (source, source_data) in enumerate(table_group):
                        with cols[col_idx]:
                            # Clean source name for display
                            display_name = source.replace('PDF_Table_', 'Table ').replace('Excel_', 'Sheet: ')
                            st.markdown(f"**{display_name}**")
                            
                            # Create display DataFrame using actual document headers
                            if not source_data.empty:
                                first_row = source_data.iloc[0]
                                
                                # Get actual headers from document
                                primary_header = first_row.get('Primary_Header', 'Column1')
                                secondary_header = first_row.get('Secondary_Header', '')
                                budget_header = first_row.get('Budget_Header', 'Amount')
                                
                                # Build display DataFrame
                                display_data = []
                                
                                for _, row in source_data.iterrows():
                                    primary_value = str(row.get(primary_header, '')).strip()
                                    secondary_value = str(row.get(secondary_header, '')).strip() if secondary_header else ''
                                    budget_amount = row.get('Budget_Amount', 0)
                                    
                                    formatted_amount = ''
                                    if budget_amount > 0:
                                        currency_symbol = row.get('Currency_Symbol', '$')
                                        formatted_amount = f"{currency_symbol}{budget_amount:,.2f}"
                                    
                                    row_data = {primary_header: primary_value, budget_header: formatted_amount}
                                    if secondary_header:
                                        row_data[secondary_header] = secondary_value
                                    
                                    display_data.append(row_data)
                                
                                # Create DataFrame with proper column order
                                if secondary_header:
                                    column_order = [primary_header, secondary_header, budget_header]
                                else:
                                    column_order = [primary_header, budget_header]
                                
                                display_df = pd.DataFrame(display_data)[column_order]
                                
                                # Clean empty values
                                for col in display_df.columns:
                                    if col != budget_header:
                                        display_df[col] = display_df[col].replace(['nan', 'NaN', 'NULL', 'null', 'None', 'N/A', 'n/a'], '')
                                
                                st.dataframe(display_df, use_container_width=True, height=200)
                            else:
                                st.write("No data to display")
                            
                            # Download button
                            csv_data = source_data.to_csv(index=False)
                            st.download_button(
                                label="CSV",
                                data=csv_data,
                                file_name=f"budget_data_{display_name.replace(' ', '_').replace(':', '')}.csv",
                                mime="text/csv",
                                key=f"download_{source}"
                            )
                
                # Display highest budget
                extractor = BudgetExtractor()
                all_data = pd.concat([data for _, data in final_tables], ignore_index=True) if final_tables else pd.DataFrame()
                
                if not all_data.empty and 'Budget_Amount' in all_data.columns:
                    highest_budget_row = all_data.loc[all_data['Budget_Amount'].idxmax()]
                    highest_amount = highest_budget_row['Budget_Amount']
                    
                    if highest_amount > 0:
                        st.markdown("---")
                        st.markdown("### Highest Budget Item")
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            label1 = highest_budget_row.get('Primary_Header', '')
                            label2 = highest_budget_row.get('Secondary_Header', '')
                            display_info = f"**{label1} - {label2}**" if label1 and label2 else f"**{label1 or label2}**"
                            st.markdown(display_info)
                        
                        with col2:
                            formatted_amount = extractor._standardize_currency_display(highest_amount)
                            st.markdown(f"**{formatted_amount}**")
            else:
                st.warning("No budget data extracted from the file.")
        else:
            st.error(f"Error: {result.get('error', 'Unknown error occurred')}")
    else:
        # Multiple files results
        st.subheader("Combined Budget Data from Multiple Files")
        # Add multi-file display logic here if needed


def handle_budget_extraction(selected_blob_file=None, blob_manager=None, table_extractor=None):
    """Handle the budget extraction process in Streamlit."""
    try:
        # If parameters are provided, process the selected file
        if selected_blob_file and blob_manager:
            st.subheader("🔄 Processing Budget Extraction")
            
            with st.spinner(f"Extracting budget data from {selected_blob_file}..."):
                # Initialize BudgetExtractor
                budget_extractor = BudgetExtractor()
                
                # Extract budget data from the selected file
                result = budget_extractor.extract_budget_data(file_path=selected_blob_file)
                
                # Display the results
                if result.get('success', False):
                    display_budget_results(result, single_file=True)
                else:
                    st.error(f"Failed to extract budget data: {result.get('error', 'Unknown error')}")
        
        # Fallback to session state handling (for uploaded files)
        elif 'uploaded_files' in st.session_state and st.session_state.uploaded_files:
            files_data = st.session_state.uploaded_files
            
            if len(files_data) == 1:
                # Single file
                file_info = files_data[0]
                result = st.session_state.get('extraction_result')
                if result:
                    display_budget_results(result, single_file=True)
            else:
                # Multiple files
                results = st.session_state.get('extraction_results', [])
                if results:
                    for result in results:
                        if result.get('success'):
                            st.subheader(f"Results from: {result.get('file_name', 'Unknown')}")
                            display_budget_results(result, single_file=True)
        else:
            st.info("Please select a file to extract budget data from.")
                    
    except Exception as e:
        st.error(f"Error in budget extraction: {str(e)}")
