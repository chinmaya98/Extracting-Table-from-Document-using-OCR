"""
AI-Enhanced Budget Extractor
Uses GPT 4.1 to intelligently process structured data from Azure Document Intelligence
"""
import os
import json
import pandas as pd
import re
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from openai import AzureOpenAI


class AIBudgetExtractor:
    """AI-powered budget table processor using GPT 4.1 for Document Intelligence data."""
    
    def __init__(self):
        load_dotenv()
        self._initialize_openai_client()
    
    def _initialize_openai_client(self):
        """Initialize Azure OpenAI client."""
        try:
            self.openai_client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION")
            )
            self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
        except Exception as e:
            print(f"Warning: Could not initialize OpenAI client: {e}")
            self.openai_client = None
            self.deployment_name = None
    
    def process_document_intelligence_data(self, tables: List[pd.DataFrame], text: str) -> Dict:
        """
        Process extracted tables and text using AI to identify and format budget data.
        
        Args:
            tables: List of DataFrames extracted by Document Intelligence
            text: Extracted text/paragraphs from Document Intelligence
            
        Returns:
            Dictionary containing processed budget data and metadata
        """
        if not self.openai_client:
            return self._fallback_processing(tables, text)
        
        try:
            # Step 1: AI identifies budget-relevant tables
            budget_tables = self._identify_budget_tables_with_ai(tables, text)
            
            if not budget_tables:
                return {
                    'success': False,
                    'error': 'No budget-related tables found by AI analysis',
                    'extracted_data': pd.DataFrame()
                }
            
            # Step 2: AI processes each budget table into 3-column format
            processed_tables = []
            for i, table in enumerate(budget_tables):
                processed_table = self._ai_process_budget_table(table, f"Table_{i+1}")
                if processed_table is not None and not processed_table.empty:
                    processed_tables.append(processed_table)
            
            # Step 3: Combine and finalize with AI
            if processed_tables:
                combined_data = pd.concat(processed_tables, ignore_index=True)
                final_data = self._ai_final_formatting(combined_data, text)
                
                return {
                    'success': True,
                    'extracted_data': final_data,
                    'tables_processed': len(budget_tables),
                    'total_items': len(final_data)
                }
            else:
                return {
                    'success': False,
                    'error': 'AI failed to process any budget tables',
                    'extracted_data': pd.DataFrame()
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"AI processing failed: {str(e)}",
                'extracted_data': pd.DataFrame()
            }
            
    def _identify_budget_tables_with_ai(self, tables: List[pd.DataFrame], text: str) -> List[pd.DataFrame]:
        """Use AI to identify which tables contain budget-related information."""
        budget_tables = []
        
        for i, table in enumerate(tables):
            if table.empty:
                continue
                
            # Convert table to string representation for AI analysis
            table_str = self._table_to_string(table)
            
            prompt = f"""
            Analyze this table extracted from a document and determine if it contains budget-related information.
            
            Table data:
            {table_str}
            
            Context from document:
            {text[:800]}...
            
            Determine if this table is budget-related by checking for:
            - Financial amounts, costs, prices, monetary values
            - Budget categories, expense items, revenue entries
            - Financial planning, allocation, or accounting data
            - Any currency symbols or financial terminology
            
            Respond with JSON:
            {{
                "is_budget_related": true/false,
                "confidence": 0.0-1.0,
                "reason": "brief explanation"
            }}
            """
            
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You are an expert financial analyst who identifies budget-related tables."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=300
                )
                
                result = json.loads(response.choices[0].message.content)
                
                if result.get("is_budget_related", False) and result.get("confidence", 0) > 0.7:
                    budget_tables.append(table)
                    print(f"Table {i+1} identified as budget-related: {result.get('reason', '')}")
                    
            except Exception as e:
                print(f"AI analysis failed for table {i}: {e}")
                # Fallback to basic check
                if self._basic_budget_check(table):
                    budget_tables.append(table)
        
        return budget_tables
    
    def _ai_process_budget_table(self, table: pd.DataFrame, table_name: str) -> Optional[pd.DataFrame]:
        """Use AI to convert a budget table preserving original headers."""
        table_str = self._table_to_string(table)
        original_headers = list(table.columns)
        
        prompt = f"""
        Convert this budget table preserving the ORIGINAL headers from the document.
        
        Original table ({table_name}):
        {table_str}
        
        Original headers from document: {original_headers}
        
        CRITICAL HEADER RULES:
        1. USE THE EXACT ORIGINAL HEADERS from the document: {original_headers}
        2. DO NOT use generic names like "Label", "Description", "Amount"
        3. Make headers BOLD using **Header Name** format
        4. Preserve the exact column names from the source document
        5. If table has 3+ columns, use the first 3 most relevant columns
        6. If less than 3 columns, duplicate the most appropriate column for missing ones
        
        FORMATTING RULES:
        1. Extract only budget-related rows (skip empty rows)
        2. Keep original header names exactly as they appear in document
        3. Clean monetary values as numbers (remove currency symbols)
        4. **Bold totals**: Mark any Total/Grand Total rows appropriately
        5. Remove completely empty rows
        6. Keep original order unless totals should be at bottom
        
        Return JSON format using ORIGINAL headers:
        {{
            "success": true,
            "headers": ["{original_headers[0] if original_headers else 'Column1'}", "{original_headers[1] if len(original_headers) > 1 else original_headers[0] if original_headers else 'Column2'}", "{original_headers[2] if len(original_headers) > 2 else original_headers[-1] if original_headers else 'Column3'}"],
            "data": [
                {{"{original_headers[0] if original_headers else 'Column1'}": "Office Supplies", "{original_headers[1] if len(original_headers) > 1 else original_headers[0] if original_headers else 'Column2'}": "Monthly supplies", "{original_headers[2] if len(original_headers) > 2 else original_headers[-1] if original_headers else 'Column3'}": 1500.00}}
            ]
        }}
        
        If cannot process: {{"success": false, "reason": "explanation"}}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert at preserving original document headers while converting financial tables. Always use the exact original column headers from the source document."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            result = json.loads(response.choices[0].message.content)
            
            if result.get("success", False) and result.get("data"):
                df = pd.DataFrame(result["data"])
                
                # Apply bold formatting to headers
                if result.get("headers"):
                    original_headers = result["headers"]
                    # Make headers bold by updating the first row or creating header info
                    bold_headers = [f"**{header}**" for header in original_headers]
                    
                    # Ensure we have the right column names
                    if len(df.columns) == len(original_headers):
                        df.columns = original_headers
                    
                    # Add a header row at the top with bold formatting
                    header_row = {col: f"**{col}**" for col in df.columns}
                    df = pd.concat([pd.DataFrame([header_row]), df], ignore_index=True)
                
                print(f"AI processed {table_name}: {len(df)} items with original headers: {original_headers if result.get('headers') else 'default'}")
                return df
            else:
                print(f"AI processing failed for {table_name}: {result.get('reason', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"AI processing error for {table_name}: {e}")
            return None
    
    def _ai_final_formatting(self, budget_data: pd.DataFrame, text: str) -> pd.DataFrame:
        """Use AI for final formatting and validation preserving original headers."""
        if budget_data.empty:
            return budget_data
        
        # Get original headers from the data
        original_headers = list(budget_data.columns)
        
        # Convert DataFrame to JSON for AI processing
        data_json = budget_data.to_json(orient='records', indent=2)
        
        prompt = f"""
        Final review and formatting of extracted budget data, preserving ORIGINAL headers.
        
        Current data with original headers {original_headers}:
        {data_json}
        
        Document context:
        {text[:600]}...
        
        CRITICAL: PRESERVE ORIGINAL HEADERS: {original_headers}
        
        Apply these formatting rules:
        1. **KEEP ORIGINAL HEADERS**: Use exact column names: {original_headers}
        2. Remove exact duplicates
        3. Ensure amounts are valid numbers (remove any remaining currency symbols)  
        4. Standardize content (proper capitalization, remove extra spaces)
        5. Enhance content using document context where helpful
        6. Make sure header row uses **bold** formatting for column names
        7. Ensure totals/subtotals use **bold** formatting in appropriate fields
        8. Sort logically (totals at bottom, otherwise by amount descending)
        9. Remove any non-budget entries that slipped through
        10. Clean up any formatting issues
        
        Return JSON with original headers:
        {{
            "success": true,
            "headers": {original_headers},
            "data": [
                {{"{original_headers[0]}": "**{original_headers[0]}**", "{original_headers[1] if len(original_headers) > 1 else original_headers[0]}": "**{original_headers[1] if len(original_headers) > 1 else original_headers[0]}**", "{original_headers[2] if len(original_headers) > 2 else original_headers[-1]}": "**{original_headers[2] if len(original_headers) > 2 else original_headers[-1]}**"}},
                {{"{original_headers[0]}": "Office Equipment", "{original_headers[1] if len(original_headers) > 1 else original_headers[0]}": "Computer hardware", "{original_headers[2] if len(original_headers) > 2 else original_headers[-1]}": 5000.00}}
            ]
        }}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert financial data formatter who preserves original document headers while ensuring clean, accurate budget data with bold header formatting."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            result = json.loads(response.choices[0].message.content)
            
            if result.get("success", False) and result.get("data"):
                formatted_df = pd.DataFrame(result["data"])
                
                # Ensure we maintain original headers
                if result.get("headers"):
                    expected_headers = result["headers"]
                    if len(formatted_df.columns) == len(expected_headers):
                        formatted_df.columns = expected_headers
                
                print(f"AI final formatting complete: {len(formatted_df)} items with headers: {list(formatted_df.columns)}")
                return formatted_df
            else:
                print("AI final formatting failed, returning original data")
                return budget_data
                
        except Exception as e:
            print(f"Final formatting error: {e}")
            return budget_data            # Use AI to enhance the extraction
            ai_enhanced_data = self._ai_enhance_budget_extraction(raw_data)
            
            if ai_enhanced_data:
                return {
                    'success': True,
                    'extracted_data': ai_enhanced_data,
                    'message': 'AI-enhanced budget extraction completed'
                }
            else:
                return basic_result
                
        except Exception as e:
            return {
                'success': False,
                'extracted_data': pd.DataFrame(),
                'error': f"AI extraction failed: {str(e)}"
            }
    
    def _get_raw_extraction_data(self, file_bytes: bytes, file_extension: str) -> Optional[Dict]:
        """Get raw extraction data (tables and text) for AI processing."""
        try:
            if file_extension.lower() in ['.xlsx', '.xls']:
                # For Excel files, get sheet data
                excel_data = pd.ExcelFile(file_bytes)
                sheets_data = []
                for sheet_name in excel_data.sheet_names:
                    df = pd.read_excel(excel_data, sheet_name=sheet_name)
                    if not df.empty:
                        sheets_data.append({
                            'sheet_name': sheet_name,
                            'data': df.to_string(),
                            'json_data': df.to_dict('records')
                        })
                
                return {
                    'type': 'excel',
                    'sheets': sheets_data,
                    'text': ''
                }
            else:
                # For PDFs and images, use Document Intelligence
                from table_extractor import TableExtractor
                extractor = TableExtractor.get_extractor()
                
                if not extractor:
                    return None
                
                if file_extension.lower() == '.pdf':
                    tables = extractor.extract_from_pdf(file_bytes)
                else:
                    tables = extractor.extract_from_image(file_bytes)
                
                # Convert tables to string format for AI processing
                tables_data = []
                for i, table in enumerate(tables):
                    if not table.empty:
                        tables_data.append({
                            'table_id': f'table_{i+1}',
                            'data': table.to_string(),
                            'json_data': table.to_dict('records')
                        })
                
                return {
                    'type': 'document',
                    'tables': tables_data,
                    'text': ''
                }
                
        except Exception as e:
            print(f"Error getting raw data: {e}")
            return None
    
    def _ai_enhance_budget_extraction(self, raw_data: Dict) -> Optional[pd.DataFrame]:
        """Use AI to enhance budget extraction."""
        if not self.openai_client:
            return None
        
        try:
            # Prepare prompt for AI
            prompt = self._create_ai_extraction_prompt(raw_data)
            
            # Call GPT 4.1
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert financial data analyst. Your task is to extract budget-related tables from documents and preserve their ORIGINAL headers with bold formatting.

Rules:
1. Extract ONLY budget-related tables (expenses, costs, financial allocations)
2. PRESERVE original column headers exactly as they appear in the document
3. Make headers BOLD using **Header Name** format
4. Remove empty rows and columns
5. Format amounts as numbers (remove currency symbols for processing)
6. Keep original header names - do NOT change to generic names like Label/Description/Amount
7. If original table has 3+ columns, use the 3 most relevant ones
8. If less than 3 columns, duplicate or adapt as needed but keep original naming pattern
9. Return data in JSON format preserving original column names"""
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=4000
            )
            
            # Parse AI response
            ai_result = self._parse_ai_response(response.choices[0].message.content)
            return ai_result
            
        except Exception as e:
            print(f"AI enhancement error: {e}")
            return None
    
    def _create_ai_extraction_prompt(self, raw_data: Dict) -> str:
        """Create prompt for AI budget extraction."""
        prompt = "Extract budget-related tables from the following data and format them into a 3-column layout:\n\n"
        
        if raw_data['type'] == 'excel':
            prompt += "EXCEL SHEETS DATA:\n"
            for sheet in raw_data['sheets']:
                prompt += f"\n--- Sheet: {sheet['sheet_name']} ---\n"
                prompt += sheet['data'][:2000]  # Limit size
                
        elif raw_data['type'] == 'document':
            prompt += "DOCUMENT TABLES DATA:\n"
            for table in raw_data['tables']:
                prompt += f"\n--- {table['table_id']} ---\n"
                prompt += table['data'][:2000]  # Limit size
        
        prompt += """\n\nPlease:
1. Extract only budget-related tables (ignore non-financial data)
2. PRESERVE ORIGINAL HEADERS from the document exactly as they appear
3. Remove empty rows and columns
4. Make original headers bold using **Header Name** format
5. Make Total/Grand Total entries bold in appropriate fields
6. DO NOT use generic column names like Label/Description/Amount
7. Keep the exact column names from the source document
8. Return as JSON array preserving original headers

Example output format (using actual document headers):
[
  {"Campaign": "**Campaign**", "Budget": "**Budget**", "Status": "**Status**"},
  {"Campaign": "Digital Marketing", "Budget": 15000.50, "Status": "Active"},
  {"Campaign": "**TOTAL**", "Budget": 50000.00, "Status": "Summary"}
]

CRITICAL: Use the EXACT original column headers from the document, not generic names."""
        
        return prompt
    
    def _parse_ai_response(self, response_text: str) -> Optional[pd.DataFrame]:
        """Parse AI response and convert to DataFrame, preserving original headers."""
        try:
            # Extract JSON from response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                return None
            
            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)
            
            if not data:
                return None
            
            # Convert to DataFrame preserving original column names
            df = pd.DataFrame(data)
            
            if df.empty:
                return None
                
            # Get the original headers (first row might be headers with ** formatting)
            original_columns = list(df.columns)
            
            # Clean numeric columns (likely amounts)
            for col in df.columns:
                # Try to convert columns that might contain amounts to numeric
                if df[col].dtype == 'object':
                    # Check if column contains numeric data
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    if not numeric_series.isna().all():  # If at least some values are numeric
                        df[col] = numeric_series.fillna(df[col])  # Keep original where conversion failed
            
            print(f"Parsed AI response with original headers: {original_columns}")
            return df
            
        except Exception as e:
            print(f"Error parsing AI response: {e}")
            return None
    
    def format_budget_table_with_formatting(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply bold formatting to headers and totals, preserving original column names."""
        if df.empty:
            return df
        
        formatted_df = df.copy()
        
        # Process each row to apply bold formatting
        for idx, row in formatted_df.iterrows():
            for col in formatted_df.columns:
                cell_value = str(row[col]).strip()
                
                # Check for headers (values that match column names or are in ** format)
                if cell_value.upper() == col.upper() or (cell_value.startswith('**') and cell_value.endswith('**')):
                    formatted_df.at[idx, col] = f"**{col}**" if not cell_value.startswith('**') else cell_value
                
                # Check for totals in any column
                elif any(keyword in cell_value.upper() for keyword in ['TOTAL', 'GRAND TOTAL', 'SUM', 'SUBTOTAL']):
                    formatted_df.at[idx, col] = f"**{cell_value}**" if not cell_value.startswith('**') else cell_value
                
                # Check for section headers (all caps with significant length)
                elif cell_value.isupper() and len(cell_value) > 2 and not cell_value.replace(' ', '').isdigit():
                    formatted_df.at[idx, col] = f"**{cell_value}**" if not cell_value.startswith('**') else cell_value
        
        return formatted_df
    
    def _table_to_string(self, table: pd.DataFrame) -> str:
        """Convert a pandas DataFrame to a string representation for AI analysis."""
        if table.empty:
            return "Empty table"
        
        # Get first few rows as string
        return table.head(10).to_string(index=False, na_rep='')
    
    def _basic_budget_check(self, table: pd.DataFrame) -> bool:
        """Basic check to see if table might contain budget data."""
        if table.empty:
            return False
        
        # Convert all data to strings and check for budget-related keywords
        table_str = table.to_string().lower()
        budget_keywords = [
            'budget', 'cost', 'price', 'amount', 'total', 'sum', 
            'expense', 'revenue', 'income', 'profit', 'loss',
            'financial', 'monetary', 'dollar', '$', '€', '£'
        ]
        
        return any(keyword in table_str for keyword in budget_keywords)
    
    def _fallback_processing(self, tables: List[pd.DataFrame]) -> pd.DataFrame:
        """Fallback method when AI processing fails - preserves original headers."""
        for table in tables:
            if self._basic_budget_check(table):
                # Preserve original headers with bold formatting
                original_headers = list(table.columns)
                
                if len(table.columns) >= 3:
                    # Use first 3 columns and keep original headers
                    result = table.iloc[:, :3].copy()
                    result.columns = original_headers[:3]
                    
                    # Add bold header row at the top
                    bold_headers = {col: f"**{col}**" for col in result.columns}
                    header_row = pd.DataFrame([bold_headers])
                    result = pd.concat([header_row, result], ignore_index=True)
                    
                elif len(table.columns) == 2:
                    # Duplicate the last column to make it 3 columns
                    result = table.copy()
                    result[original_headers[-1] + '_2'] = table.iloc[:, -1]
                    
                    # Add bold header row
                    bold_headers = {col: f"**{col}**" for col in result.columns}
                    header_row = pd.DataFrame([bold_headers])
                    result = pd.concat([header_row, result], ignore_index=True)
                    
                elif len(table.columns) == 1:
                    # Create 3 columns by duplicating the single column
                    result = table.copy()
                    result[original_headers[0] + '_desc'] = table.iloc[:, 0]
                    result[original_headers[0] + '_amount'] = table.iloc[:, 0]
                    
                    # Add bold header row
                    bold_headers = {col: f"**{col}**" for col in result.columns}
                    header_row = pd.DataFrame([bold_headers])
                    result = pd.concat([header_row, result], ignore_index=True)
                
                return result.dropna()
        
        # Return empty DataFrame with generic bold headers if no suitable table found
        generic_headers = ['**Item**', '**Description**', '**Amount**']
        return pd.DataFrame(columns=generic_headers)

    def generate_budget_summary(self, tables: List[pd.DataFrame], text: str, filename: str = "") -> str:
        """
        Generate an AI-powered budget summary when no budget tables are found.
        
        Args:
            tables: List of DataFrames from the document (even if not budget-related)
            text: Extracted text from the document
            filename: Name of the source file
            
        Returns:
            A 2-3 line summary of any budget-related content found in the document
        """
        if not self.openai_client:
            return self._generate_fallback_summary(tables, text, filename)
        
        try:
            # Prepare document content for AI analysis
            content_preview = text[:2000] if text else ""
            
            # Get table summaries
            table_summaries = []
            for i, table in enumerate(tables[:3]):  # Limit to first 3 tables
                table_str = self._table_to_string(table)
                table_summaries.append(f"Table {i+1}:\n{table_str[:500]}")
            
            tables_content = "\n\n".join(table_summaries)
            
            prompt = f"""
            You are analyzing a document for budget and financial content. Provide a comprehensive budget analysis summary.
            
            Document: {filename}
            
            Text Content:
            {content_preview}
            
            Tables Found:
            {tables_content}
            
            TASK: Analyze this document and provide a detailed 2-3 line summary focusing SPECIFICALLY on budget and financial information.
            
            Look for and mention:
            ✓ Specific budget amounts, costs, expenses, revenue figures
            ✓ Budget categories (marketing, operations, salaries, etc.)
            ✓ Financial planning, allocations, or projections
            ✓ Any monetary values, pricing, or cost information
            ✓ Budget periods (monthly, quarterly, annual)
            ✓ Financial targets or goals
            
            IMPORTANT: Even if this document doesn't contain formal budget tables, look for ANY financial or monetary information and describe it. If there are numbers that could be costs, prices, or amounts, mention them.
            
            If absolutely no financial content exists, describe what the document IS about, but still frame it in terms of potential budget relevance.
            
            Format: Write exactly 2-3 informative lines that give the user valuable budget insights about this document.
            """
            
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a financial analyst who summarizes budget and financial content from documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
            
        except Exception as e:
            print(f"AI summary generation failed: {e}")
            return self._generate_fallback_summary(tables, text, filename)
    
    def _generate_fallback_summary(self, tables: List[pd.DataFrame], text: str, filename: str = "") -> str:
        """Generate a basic summary when AI is unavailable."""
        try:
            # Basic analysis without AI
            total_tables = len(tables)
            total_text_length = len(text) if text else 0
            
            # Look for financial keywords in text
            financial_keywords = ['budget', 'cost', 'price', 'amount', 'expense', 'revenue', '$', '€', '£', 'financial']
            keyword_matches = []
            
            if text:
                text_lower = text.lower()
                keyword_matches = [kw for kw in financial_keywords if kw in text_lower]
            
            # Basic summary
            if keyword_matches:
                summary = f"Document '{filename}' contains {total_tables} table(s) with financial references including: {', '.join(keyword_matches[:3])}. "
                summary += f"While no structured budget tables were found, the document appears to contain financial information. "
                summary += f"Manual review may reveal budget-related data that requires different formatting."
            else:
                summary = f"Document '{filename}' contains {total_tables} table(s) but no clear budget or financial data was detected. "
                summary += f"The content may be non-financial in nature or use terminology not recognized by the automatic detection system."
            
            return summary
            
        except Exception:
            return f"Unable to analyze content from '{filename}'. The document was processed but no budget data could be extracted."


def test_ai_budget_extractor():
    """Test the AI budget extractor."""
    extractor = AIBudgetExtractor()
    
    # Test with a sample file (you can modify this path)
    test_file_path = "sample_budget.xlsx"  # Replace with actual test file
    
    if os.path.exists(test_file_path):
        with open(test_file_path, 'rb') as f:
            file_bytes = f.read()
        
        result = extractor.extract_ai_enhanced_budget_data(file_bytes, '.xlsx')
        
        if result['success']:
            print("AI Budget Extraction Successful!")
            print(result['extracted_data'])
        else:
            print(f"Extraction failed: {result.get('error', 'Unknown error')}")
    else:
        print(f"Test file {test_file_path} not found")


if __name__ == "__main__":
    test_ai_budget_extractor()
