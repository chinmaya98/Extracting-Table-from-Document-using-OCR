"""
Budget Extraction Module
Processes extracted tables to identify and extract budget/monetary data.
"""

import pandas as pd
import re
from utils.currency_utils import contains_money, MONEY_KEYWORDS, MONEY_PATTERN


class BudgetExtractor:
    """
    Extracts budget/monetary information from extracted tables.
    """
    
    def __init__(self):
        """Initialize the budget extractor."""
        pass
    
    def extract_budget_from_tables(self, tables):
        """
        Extract budget information from a list of tables.
        
        Args:
            tables: List of (sheet_name, DataFrame) tuples or single DataFrame
            
        Returns:
            pandas.DataFrame: Two-column DataFrame with 'Description' and 'Budget' columns
        """
        if isinstance(tables, pd.DataFrame):
            # Single DataFrame
            return self._extract_budget_from_single_table(tables)
        elif isinstance(tables, list):
            # List of tables
            all_budget_data = []
            
            for table_info in tables:
                if isinstance(table_info, tuple) and len(table_info) >= 2:
                    sheet_name, df = table_info[0], table_info[1]
                    budget_data = self._extract_budget_from_single_table(df, sheet_name)
                    if not budget_data.empty:
                        all_budget_data.append(budget_data)
                elif isinstance(table_info, pd.DataFrame):
                    budget_data = self._extract_budget_from_single_table(table_info)
                    if not budget_data.empty:
                        all_budget_data.append(budget_data)
            
            if all_budget_data:
                # Combine all budget data
                combined_budget = pd.concat(all_budget_data, ignore_index=True)
                return self._select_best_budget_columns(combined_budget)
            else:
                return pd.DataFrame(columns=['Description', 'Budget'])
        else:
            return pd.DataFrame(columns=['Description', 'Budget'])
    
    def _extract_budget_from_single_table(self, df, table_name=None):
        """
        Extract budget information from a single DataFrame.
        
        Args:
            df: pandas.DataFrame
            table_name: Optional name for the table (for labeling)
            
        Returns:
            pandas.DataFrame: Budget data with Description and Budget columns
        """
        if df.empty:
            return pd.DataFrame(columns=['Description', 'Budget'])
        
        # Find columns that contain monetary data
        money_columns = self._identify_money_columns(df)
        label_columns = self._identify_label_columns(df)
        
        if not money_columns or not label_columns:
            return pd.DataFrame(columns=['Description', 'Budget'])
        
        # Select the best money column (prioritize columns with totals, then highest non-null count)
        best_money_col = self._select_best_money_column(df, money_columns)
        
        # Select the best label column (highest non-null count, preferably text)
        if len(label_columns) == 1:
            best_label_col = label_columns[0]
        else:
            # Calculate non-null counts for each label column
            label_scores = []
            for col in label_columns:
                # Handle both single and multi-index columns
                col_data = df[col]
                if hasattr(col_data, 'shape') and len(col_data.shape) > 1:
                    # Multi-column selection, take the first column
                    count = col_data.iloc[:, 0].notna().sum()
                else:
                    count = col_data.notna().sum()
                
                # Ensure we get a scalar value
                if hasattr(count, 'iloc'):
                    count = count.iloc[0] if count.size > 0 else 0
                elif hasattr(count, 'item'):
                    count = count.item() if count.size == 1 else count.sum()
                
                label_scores.append((col, count))
            
            best_label_col = max(label_scores, key=lambda x: x[1])[0]
        
        # Identify total rows to ensure they're included
        total_rows = self._identify_total_rows(df, best_money_col)
        
        # Create budget DataFrame
        budget_df = pd.DataFrame()
        
        # Handle potential multi-column selections
        label_data = df[best_label_col]
        if hasattr(label_data, 'shape') and len(label_data.shape) > 1:
            label_data = label_data.iloc[:, 0]  # Take first column
        budget_df['Description'] = label_data.astype(str)
        
        money_data = df[best_money_col]
        if hasattr(money_data, 'shape') and len(money_data.shape) > 1:
            money_data = money_data.iloc[:, 0]  # Take first column
        budget_df['Budget'] = money_data
        
        # Mark total rows for special handling
        budget_df['_is_total'] = False
        budget_df.loc[total_rows, '_is_total'] = True
        
        # Clean the data while preserving totals
        budget_df = self._clean_budget_data_with_totals(budget_df)
        
        # Add table name prefix if provided
        if table_name:
            budget_df['Description'] = budget_df['Description'].apply(
                lambda x: f"{table_name}: {x}" if x and str(x) != 'nan' else table_name
            )
        
        return budget_df
    
    def _select_best_money_column(self, df, money_columns):
        """
        Select the best money column, prioritizing those with total values.
        
        Args:
            df: pandas.DataFrame
            money_columns: List of potential money columns
            
        Returns:
            str: Name of the best money column
        """
        if len(money_columns) == 1:
            return money_columns[0]
        
        # Score each column
        column_scores = {}
        
        for col in money_columns:
            score = 0
            
            # Base score: non-null count
            notna_count = df[col].notna().sum()
            # Handle potential Series result from MultiIndex
            if hasattr(notna_count, 'iloc'):
                notna_count = notna_count.iloc[0] if notna_count.size > 0 else 0
            elif hasattr(notna_count, 'item'):
                notna_count = notna_count.item() if notna_count.size == 1 else notna_count.sum()
            score += notna_count
            
            # Bonus for having total-like values
            total_rows = self._identify_total_rows(df, col)
            score += len(total_rows) * 10  # Heavy weight for totals
            
            # Bonus for column name containing key terms
            col_str = str(col).lower()
            if any(term in col_str for term in ['total', 'amount', 'budget', 'sum']):
                score += 20
            
            column_scores[col] = score
        
        # Return the column with the highest score
        return max(column_scores.items(), key=lambda x: x[1])[0]
    
    def _identify_money_columns(self, df):
        """
        Identify columns that contain monetary data.
        
        Args:
            df: pandas.DataFrame
            
        Returns:
            List of column names that contain monetary data
        """
        money_columns = []
        
        for col in df.columns:
            col_str = str(col).lower()
            
            # Enhanced money keywords including more common terms
            extended_money_keywords = MONEY_KEYWORDS + [
                'total', 'sum', 'subtotal', 'grand total', 'net', 'gross',
                'value', 'amount', 'balance', 'price', 'cost', 'expense',
                'income', 'revenue', 'budget', 'allocation', 'fund'
            ]
            
            # Check if column name contains money keywords
            if any(keyword in col_str for keyword in extended_money_keywords):
                money_columns.append(col)
                continue
            
            # Check if column values contain monetary patterns
            has_money_pattern = False
            numeric_count = 0
            non_null_values = df[col].dropna().astype(str)
            
            for value in non_null_values.head(30):  # Check first 30 non-null values
                value_str = str(value).strip()
                
                # Check for monetary patterns
                if MONEY_PATTERN.search(value_str):
                    has_money_pattern = True
                    break
                
                # Check for numeric values (potential money without symbols)
                if self._is_likely_money_value(value_str):
                    numeric_count += 1
            
            # If most values look like money or we found money patterns
            if has_money_pattern or (numeric_count > len(non_null_values.head(30)) * 0.6):
                money_columns.append(col)
        
        return money_columns
    
    def _identify_label_columns(self, df):
        """
        Identify columns that could serve as labels.
        
        Args:
            df: pandas.DataFrame
            
        Returns:
            List of column names that could serve as labels
        """
        label_columns = []
        
        for col in df.columns:
            col_str = str(col).lower()
            
            # Skip if it's likely a money column
            if any(keyword in col_str for keyword in MONEY_KEYWORDS):
                continue
            
            # Prefer columns with text-like names
            label_keywords = ['name', 'item', 'description', 'category', 'title', 'label', 'type', 'account']
            if any(keyword in col_str for keyword in label_keywords):
                label_columns.append(col)
                continue
            
            # Check if column contains mostly text data (not numbers)
            non_null_values = df[col].dropna().astype(str)
            if len(non_null_values) > 0:
                # Count how many values are not pure numbers
                text_count = sum(1 for value in non_null_values.head(20) 
                               if not self._is_pure_number(str(value)))
                if text_count > len(non_null_values.head(20)) * 0.7:  # 70% are text
                    label_columns.append(col)
        
        # If no specific label columns found, include all non-money columns
        if not label_columns:
            money_columns = self._identify_money_columns(df)
            label_columns = [col for col in df.columns if col not in money_columns]
        
        return label_columns
    
    def _is_pure_number(self, value):
        """Check if a value is a pure number."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def _is_likely_money_value(self, value):
        """Check if a value looks like a monetary amount."""
        if not value or str(value).strip() == '':
            return False
        
        value_str = str(value).strip()
        
        # Remove common currency symbols and separators
        cleaned = re.sub(r'[$€£¥₹,\s]', '', value_str)
        
        # Check if it's a number (possibly with decimal)
        try:
            float_val = float(cleaned)
            # Reasonable range for monetary values (0 to very large numbers)
            return float_val >= 0 and len(cleaned) > 0
        except (ValueError, TypeError):
            return False
    
    def _identify_total_rows(self, df, money_col):
        """
        Identify rows that likely contain totals or subtotals.
        
        Args:
            df: pandas.DataFrame
            money_col: Column name containing monetary values
            
        Returns:
            List of row indices that likely contain totals
        """
        total_rows = []
        
        for idx, row in df.iterrows():
            # Check description column for total indicators
            for col in df.columns:
                if col != money_col:
                    cell_value = str(row[col]).lower().strip()
                    total_keywords = [
                        'total', 'sum', 'subtotal', 'grand total', 'net total',
                        'overall', 'summary', 'aggregate', 'combined'
                    ]
                    if any(keyword in cell_value for keyword in total_keywords):
                        total_rows.append(idx)
                        break
        
        return total_rows
    
    def _clean_budget_data_with_totals(self, budget_df):
        """
        Clean the budget DataFrame while preserving total rows.
        
        Args:
            budget_df: pandas.DataFrame with Description, Budget, and _is_total columns
            
        Returns:
            Cleaned pandas.DataFrame
        """
        if budget_df.empty:
            return budget_df
        
        # Separate totals from regular data
        total_rows = budget_df[budget_df['_is_total']].copy()
        regular_rows = budget_df[~budget_df['_is_total']].copy()
        
        # Clean regular rows
        if not regular_rows.empty:
            regular_rows = self._clean_budget_data(regular_rows.drop(columns=['_is_total']))
        
        # Clean total rows more leniently (keep even if description is generic)
        if not total_rows.empty:
            # Remove rows where both Description and Budget are empty/null
            total_rows = total_rows.dropna(subset=['Budget'])
            
            # Clean up Description column
            total_rows['Description'] = total_rows['Description'].astype(str).str.strip()
            
            # Clean up Budget column
            total_rows['Budget'] = total_rows['Budget'].apply(self._extract_numeric_value)
            
            # Remove rows where Budget couldn't be converted to a number
            total_rows = total_rows[total_rows['Budget'].notna()]
            
            # Drop the helper column
            total_rows = total_rows.drop(columns=['_is_total'])
        
        # Combine back together
        if not regular_rows.empty and not total_rows.empty:
            combined = pd.concat([regular_rows, total_rows], ignore_index=True)
        elif not regular_rows.empty:
            combined = regular_rows
        elif not total_rows.empty:
            combined = total_rows
        else:
            combined = pd.DataFrame(columns=['Description', 'Budget'])
        
        return combined.reset_index(drop=True)
    
    def _clean_budget_data(self, budget_df):
        """
        Clean the budget DataFrame by removing empty rows and formatting data.
        
        Args:
            budget_df: pandas.DataFrame with Description and Budget columns
            
        Returns:
            Cleaned pandas.DataFrame
        """
        # Remove rows where both Description and Budget are empty/null
        budget_df = budget_df.dropna(subset=['Description', 'Budget'], how='all')
        
        # Remove rows where Description is empty or just whitespace
        budget_df = budget_df[budget_df['Description'].astype(str).str.strip() != '']
        budget_df = budget_df[budget_df['Description'].astype(str) != 'nan']
        
        # Clean up Description column
        budget_df['Description'] = budget_df['Description'].astype(str).str.strip()
        
        # Clean up Budget column - try to extract numeric values
        budget_df['Budget'] = budget_df['Budget'].apply(self._extract_numeric_value)
        
        # Remove rows where Budget couldn't be converted to a number
        budget_df = budget_df[budget_df['Budget'].notna()]
        
        return budget_df.reset_index(drop=True)
    
    def _extract_numeric_value(self, value):
        """
        Extract numeric value from a string that might contain currency symbols.
        Enhanced to handle Excel formatting and various currency formats.
        
        Args:
            value: String or numeric value
            
        Returns:
            Float value or None if no numeric value found
        """
        if pd.isna(value):
            return None
        
        value_str = str(value).strip()
        
        # If already a number, return it
        try:
            return float(value_str)
        except (ValueError, TypeError):
            pass
        
        # Handle Excel's scientific notation (e.g., 1.23E+5)
        if 'e+' in value_str.lower() or 'e-' in value_str.lower():
            try:
                return float(value_str)
            except (ValueError, TypeError):
                pass
        
        # Remove currency symbols, spaces, and common separators
        # Handle various currency formats: $1,234.56, €1.234,56, ₹1,23,456.78, etc.
        cleaned = re.sub(r'[$€£¥₹₦₦¢\s]', '', value_str)
        
        # Handle parentheses for negative numbers (accounting format)
        is_negative = False
        if cleaned.startswith('(') and cleaned.endswith(')'):
            cleaned = cleaned[1:-1]
            is_negative = True
        
        # Handle different decimal separators and thousand separators
        # European format: 1.234.567,89 or 1 234 567,89
        if ',' in cleaned and '.' in cleaned:
            # If comma comes after dot, it's likely decimal separator
            if cleaned.rfind(',') > cleaned.rfind('.'):
                # European format: 1.234,56
                cleaned = cleaned.replace('.', '').replace(',', '.')
            else:
                # US format: 1,234.56
                cleaned = cleaned.replace(',', '')
        elif ',' in cleaned and not '.' in cleaned:
            # Could be thousand separator or decimal separator
            comma_pos = cleaned.rfind(',')
            # If there are exactly 2 digits after the last comma, it's likely decimal
            if len(cleaned) - comma_pos - 1 == 2:
                cleaned = cleaned.replace(',', '.')
            else:
                cleaned = cleaned.replace(',', '')
        
        # Try to extract numbers from the cleaned string
        numeric_match = re.search(r'\d+\.?\d*', cleaned)
        
        if numeric_match:
            try:
                result = float(numeric_match.group())
                return -result if is_negative else result
            except (ValueError, TypeError):
                pass
        
        return None
    
    def _select_best_budget_columns(self, combined_budget):
        """
        Select the best budget data if multiple tables were processed.
        Prioritizes keeping total rows and most relevant data.
        
        Args:
            combined_budget: Combined DataFrame from multiple tables
            
        Returns:
            pandas.DataFrame: Best budget data with totals preserved
        """
        if combined_budget.empty:
            return combined_budget
        
        # Identify potential total rows based on description content
        total_mask = combined_budget['Description'].str.lower().str.contains(
            r'\b(?:total|sum|subtotal|grand total|net total|overall|summary)\b', 
            regex=True, na=False
        )
        
        total_rows = combined_budget[total_mask].copy()
        regular_rows = combined_budget[~total_mask].copy()
        
        # Always keep all total rows
        result_rows = [total_rows] if not total_rows.empty else []
        
        # For regular rows, if too many, select the most relevant ones
        if not regular_rows.empty:
            if len(regular_rows) > 40:  # Leave room for totals
                # Sort by budget amount (descending) and take top 40
                regular_rows = regular_rows.sort_values('Budget', ascending=False).head(40)
            result_rows.append(regular_rows)
        
        # Combine and sort: totals at the end
        if result_rows:
            final_result = pd.concat(result_rows, ignore_index=True)
            
            # Sort so that totals appear at the bottom
            total_mask_final = final_result['Description'].str.lower().str.contains(
                r'\b(?:total|sum|subtotal|grand total|net total|overall|summary)\b', 
                regex=True, na=False
            )
            
            regular_final = final_result[~total_mask_final]
            totals_final = final_result[total_mask_final]
            
            # Sort regular rows by budget amount (descending)
            if not regular_final.empty:
                regular_final = regular_final.sort_values('Budget', ascending=False)
            
            # Combine with totals at the end
            if not regular_final.empty and not totals_final.empty:
                final_result = pd.concat([regular_final, totals_final], ignore_index=True)
            elif not regular_final.empty:
                final_result = regular_final
            else:
                final_result = totals_final
            
            return final_result.reset_index(drop=True)
        
        return pd.DataFrame(columns=['Description', 'Budget'])


def get_budget_extractor():
    """Factory function to get a BudgetExtractor instance."""
    return BudgetExtractor()
