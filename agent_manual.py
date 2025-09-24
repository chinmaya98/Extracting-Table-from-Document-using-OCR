"""
Manual Review Agent for Budget Extraction
Performs AI-powered manual document review when no structured budget tables are found.
Uses the same GPT-4.1 model to analyze unstructured financial data.
"""
import os
import json
import pandas as pd
import re
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import AzureOpenAI
import streamlit as st


class ManualReviewAgent:
    """
    Manual review agent that performs AI-powered document analysis
    when no structured budget tables are found.
    """
    
    def __init__(self):
        load_dotenv()
        self._initialize_openai_client()
    
    def _initialize_openai_client(self):
        """Initialize Azure OpenAI client using the same configuration as main extractor."""
        try:
            self.openai_client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
            )
            self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
            print(f"Manual Review Agent initialized with deployment: {self.deployment_name}")
        except Exception as e:
            print(f"Warning: Could not initialize Manual Review Agent OpenAI client: {e}")
            self.openai_client = None
            self.deployment_name = None
    
    def perform_manual_review(self, text: str, tables: List[pd.DataFrame], filename: str = "") -> Dict:
        """
        Perform manual AI-powered review of document content to extract budget data.
        
        Args:
            text: Extracted text from the document
            tables: List of DataFrames (may be empty or non-budget related)
            filename: Original filename for context
            
        Returns:
            Dictionary containing manually extracted budget data with warnings
        """
        if not self.openai_client:
            return self._fallback_manual_review(text, tables, filename)
        
        try:
            # Store original tables for validation
            self.original_ocr_tables = tables
            
            # Prepare context for AI analysis
            context = self._prepare_analysis_context(text, tables, filename)
            
            # Perform AI-powered manual review
            ai_response = self._call_ai_for_manual_review(context)
            
            # Parse and validate the AI response
            extracted_data = self._parse_ai_response(ai_response)
            
            # Create structured output with warnings and OCR validation
            result = self._create_manual_review_result(extracted_data, filename, tables)
            
            return result
            
        except Exception as e:
            print(f"Error in manual review: {e}")
            return self._fallback_manual_review(text, tables, filename)
    
    def _prepare_analysis_context(self, text: str, tables: List[pd.DataFrame], filename: str) -> str:
        """Prepare comprehensive context for AI analysis."""
        context = f"""
MANUAL DOCUMENT REVIEW REQUEST
Filename: {filename}

TASK: Analyze this document for any financial/budget information that may not be in structured table format.

DOCUMENT TEXT:
{text[:8000]}  # Limit text to prevent token overflow

EXISTING TABLES (if any):
"""
        
        # Include information about existing tables
        if tables:
            for i, table in enumerate(tables[:3]):  # Limit to first 3 tables
                context += f"\nTable {i+1}:\n"
                context += f"Columns: {list(table.columns)}\n"
                context += f"Sample data:\n{table.head(3).to_string()}\n"
        else:
            context += "No structured tables found.\n"
        
        return context
    
    def _call_ai_for_manual_review(self, context: str) -> str:
        """Call GPT-4.1 to perform manual document review."""
        
        system_prompt = """
You are a financial document analysis expert. Your task is to manually review document content and extract any budget, financial, or cost-related information, even if it's not in a structured table format.

INSTRUCTIONS:
1. Look for ANY financial information: budgets, costs, expenses, revenues, prices, amounts, totals
2. Extract line items, categories, descriptions, and amounts
3. If amounts are mentioned without clear structure, do your best to organize them logically
4. Include context about where the information was found
5. Be explicit about uncertainty - mark unclear or potentially inaccurate extractions

OUTPUT FORMAT (JSON):
{
    "found_financial_data": true/false,
    "extracted_items": [
        {
            "category": "description or category",
            "description": "detailed description",
            "amount": "amount with currency if possible",
            "confidence": "high/medium/low",
            "source": "where in document this was found"
        }
    ],
    "summary": "brief summary of financial content found",
    "warnings": ["list of warnings about data quality/uncertainty"],
    "total_amount": "total if calculable, otherwise null"
}

Be thorough but honest about limitations. Mark uncertain data clearly.
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.3,  # Lower temperature for more consistent extraction
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"AI call failed in manual review: {e}")
            raise
    
    def _parse_ai_response(self, ai_response: str) -> Dict:
        """Parse and validate AI response."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                ai_data = json.loads(json_match.group())
            else:
                # If no JSON found, create basic structure
                ai_data = {
                    "found_financial_data": False,
                    "extracted_items": [],
                    "summary": "Could not parse AI response properly",
                    "warnings": ["AI response parsing failed"],
                    "total_amount": None
                }
            
            # Validate structure
            required_keys = ["found_financial_data", "extracted_items", "summary", "warnings"]
            for key in required_keys:
                if key not in ai_data:
                    ai_data[key] = [] if key in ["extracted_items", "warnings"] else "Not provided"
            
            return ai_data
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in manual review: {e}")
            return {
                "found_financial_data": False,
                "extracted_items": [],
                "summary": "Failed to parse AI response",
                "warnings": [f"JSON parsing error: {str(e)}"],
                "total_amount": None
            }
    
    def _create_manual_review_result(self, ai_data: Dict, filename: str, original_tables: List[pd.DataFrame] = None) -> Dict:
        """Create structured result with proper warnings and OCR validation."""
        
        # Create DataFrame from extracted items
        if ai_data.get("found_financial_data") and ai_data.get("extracted_items"):
            items = ai_data["extracted_items"]
            
            # Convert to standardized format with OCR validation
            df_data = []
            for item in items:
                row = {
                    "**Category**": item.get("category", "Unknown"),
                    "**Description**": item.get("description", ""),
                    "**Amount**": item.get("amount", "")
                }
                
                # Add confidence and source as part of description if available
                if item.get("confidence"):
                    row["**Description**"] += f" [Confidence: {item['confidence']}]"
                if item.get("source"):
                    row["**Description**"] += f" [Source: {item['source']}]"
                
                # Validate against OCR tables if available
                if original_tables:
                    validation_result = self._validate_against_ocr_tables(item, original_tables)
                    if validation_result:
                        row["**Description**"] += f" [OCR Validation: {validation_result}]"
                
                df_data.append(row)
            
            result_df = pd.DataFrame(df_data)
        else:
            # Create empty DataFrame with standard headers
            result_df = pd.DataFrame(columns=["**Category**", "**Description**", "**Amount**"])
        
        # Prepare comprehensive warnings including OCR validation info
        warnings = [
            "WARNING: MANUAL AI REVIEW - DATA MAY CONTAIN HALLUCINATIONS",
            "WARNING: Please double-check all extracted values manually",
            "WARNING: This data was extracted from unstructured text using AI analysis"
        ]
        
        # Add OCR validation info
        if original_tables and len(original_tables) > 0:
            warnings.append(f"INFO: Values cross-referenced against {len(original_tables)} OCR-extracted table(s)")
            warnings.append("INFO: Check [OCR Validation] status in description column")
        else:
            warnings.append("WARNING: No OCR tables available for cross-validation")
        
        # Add specific AI warnings
        if ai_data.get("warnings"):
            warnings.extend([f"AI Warning: {w}" for w in ai_data["warnings"]])
        
        result = {
            'success': True,
            'method': 'manual_ai_review',
            'table': result_df,
            'filename': filename,
            'summary': ai_data.get("summary", "Manual AI review completed"),
            'warnings': warnings,
            'total_items': len(result_df),
            'ai_confidence': 'mixed',  # Always mixed for manual review
            'found_financial_data': ai_data.get("found_financial_data", False),
            'total_amount': ai_data.get("total_amount"),
            'review_type': 'manual_ai_extraction'
        }
        
        return result
    
    def _validate_against_ocr_tables(self, extracted_item: Dict, ocr_tables: List[pd.DataFrame]) -> str:
        """
        Validate AI-extracted financial data against OCR-extracted tables.
        
        Args:
            extracted_item: Single item extracted by AI (dict with category, amount, etc.)
            ocr_tables: List of DataFrames extracted from OCR
            
        Returns:
            String describing validation result (Found/Not Found/Partial Match)
        """
        if not ocr_tables or not extracted_item:
            return "No OCR data to validate against"
        
        extracted_amount = extracted_item.get("amount", "").strip()
        extracted_category = extracted_item.get("category", "").strip().lower()
        
        # Clean amount for comparison (remove currency symbols, spaces)
        clean_amount = re.sub(r'[^\d,.]', '', extracted_amount)
        
        validation_results = []
        
        for table_idx, table in enumerate(ocr_tables):
            if table.empty:
                continue
            
            # Convert table to string for searching
            table_str = table.to_string().lower()
            
            # Check for amount matches
            amount_found = False
            if clean_amount and len(clean_amount) > 2:  # Only check meaningful amounts
                # Look for exact amount match
                amount_patterns = [
                    clean_amount,
                    clean_amount.replace(',', ''),
                    clean_amount.replace('.', ''),
                ]
                
                for pattern in amount_patterns:
                    if pattern and pattern in table_str:
                        amount_found = True
                        break
            
            # Check for category/description matches (fuzzy matching)
            category_found = False
            if extracted_category and len(extracted_category) > 3:
                # Split category into words and check for partial matches
                category_words = extracted_category.split()
                word_matches = 0
                for word in category_words:
                    if len(word) > 3 and word in table_str:  # Only check meaningful words
                        word_matches += 1
                
                if word_matches > 0:
                    category_found = True
            
            # Determine validation result for this table
            if amount_found and category_found:
                validation_results.append(f"Full match in Table {table_idx + 1}")
            elif amount_found:
                validation_results.append(f"Amount found in Table {table_idx + 1}")
            elif category_found:
                validation_results.append(f"Category match in Table {table_idx + 1}")
        
        # Return best validation result
        if validation_results:
            return validation_results[0]  # Return the first/best match
        else:
            return "Not found in OCR tables - Possible hallucination"
    
    def _fallback_manual_review(self, text: str, tables: List[pd.DataFrame], filename: str) -> Dict:
        """Fallback when AI is not available - basic text analysis."""
        
        # Look for currency symbols and numbers in text
        currency_patterns = [
            r'\$[\d,]+\.?\d*',  # Dollar amounts
            r'€[\d,]+\.?\d*',   # Euro amounts
            r'£[\d,]+\.?\d*',   # Pound amounts
            r'[\d,]+\.?\d*\s*(USD|EUR|GBP|dollars?|euros?|pounds?)',  # Amounts with currency words
        ]
        
        found_amounts = []
        for pattern in currency_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found_amounts.extend(matches)
        
        # Create basic result
        if found_amounts:
            df_data = []
            for i, amount in enumerate(found_amounts[:20]):  # Limit to 20 items
                df_data.append({
                    "**Category**": f"Financial Item {i+1}",
                    "**Description**": "Amount found in document text (fallback extraction)",
                    "**Amount**": str(amount)
                })
            
            result_df = pd.DataFrame(df_data)
            found_data = True
        else:
            result_df = pd.DataFrame(columns=["**Category**", "**Description**", "**Amount**"])
            found_data = False
        
        warnings = [
            "WARNING: FALLBACK MANUAL REVIEW - AI NOT AVAILABLE",
            "WARNING: Basic pattern matching used - accuracy not guaranteed",
            "WARNING: Please perform complete manual review of the document"
        ]
        
        return {
            'success': True,
            'method': 'fallback_manual_review',
            'table': result_df,
            'filename': filename,
            'summary': f"Fallback review found {len(found_amounts)} potential financial references",
            'warnings': warnings,
            'total_items': len(result_df),
            'ai_confidence': 'none',
            'found_financial_data': found_data,
            'total_amount': None,
            'review_type': 'fallback_pattern_matching'
        }
    
    def display_manual_review_results(self, result: Dict):
        """Display manual review results in Streamlit with appropriate warnings."""
        
        st.warning("Manual AI Review Performed - No structured budget tables found")
        
        # Display warnings prominently
        if result.get('warnings'):
            for warning in result['warnings']:
                if warning.startswith("INFO:"):
                    st.info(warning)
                else:
                    st.warning(warning)
        
        st.info(f"**Review Summary:** {result.get('summary', 'Manual review completed')}")
        
        # Display extracted data
        if result.get('found_financial_data') and not result['table'].empty:
            st.success(f"Found {result.get('total_items', 0)} financial items through manual AI analysis")
            
            st.subheader("Manually Extracted Financial Data")
            
            # Add explanation about OCR validation
            st.info("💡 **OCR Validation Legend**: Each item is cross-referenced with original OCR tables to detect potential hallucinations.")
            
            st.dataframe(result['table'], use_container_width=True)
            
            if result.get('total_amount'):
                st.metric("Total Amount (if calculable)", result['total_amount'])
                
            # Show OCR validation summary
            table_data = result['table']
            if not table_data.empty:
                validation_stats = self._get_validation_statistics(table_data)
                if validation_stats:
                    st.subheader("OCR Validation Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Verified in OCR", validation_stats.get('verified', 0))
                    with col2:
                        st.metric("Partial Matches", validation_stats.get('partial', 0))
                    with col3:
                        st.metric("Potential Hallucinations", validation_stats.get('hallucinations', 0))
        else:
            st.error("No financial data could be extracted through manual review")
        
        # Display metadata
        with st.expander("Manual Review Details"):
            st.write(f"**Review Method:** {result.get('method', 'Unknown')}")
            st.write(f"**Review Type:** {result.get('review_type', 'Unknown')}")
            st.write(f"**AI Confidence:** {result.get('ai_confidence', 'Unknown')}")
            st.write(f"**File Analyzed:** {result.get('filename', 'Unknown')}")
    
    def _get_validation_statistics(self, table_data: pd.DataFrame) -> Dict:
        """Extract validation statistics from the processed table."""
        stats = {'verified': 0, 'partial': 0, 'hallucinations': 0}
        
        for _, row in table_data.iterrows():
            description = str(row.get('**Description**', ''))
            
            if '[OCR Validation:' in description:
                if 'Full match' in description or 'Amount found' in description:
                    stats['verified'] += 1
                elif 'Category match' in description:
                    stats['partial'] += 1
                elif 'Not found' in description or 'hallucination' in description:
                    stats['hallucinations'] += 1
        
        return stats


# Helper function for integration with main application
def trigger_manual_review(text: str, tables: List[pd.DataFrame], filename: str = "") -> Dict:
    """
    Trigger manual review agent when no structured budget data is found.
    
    Args:
        text: Extracted text from document
        tables: List of DataFrames (may be empty)
        filename: Original filename
        
    Returns:
        Dictionary containing manual review results
    """
    agent = ManualReviewAgent()
    return agent.perform_manual_review(text, tables, filename)


# Streamlit integration function
def display_manual_review_ui(result: Dict):
    """Display manual review results in Streamlit UI."""
    agent = ManualReviewAgent()
    agent.display_manual_review_results(result)