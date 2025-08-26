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
        
        # Select the best money column (highest non-null count)
        best_money_col = max(money_columns, key=lambda col: df[col].notna().sum())
        
        # Select the best label column (highest non-null count, preferably text)
        best_label_col = max(label_columns, key=lambda col: df[col].notna().sum())
        
        # Create budget DataFrame
        budget_df = pd.DataFrame()
        budget_df['Description'] = df[best_label_col].astype(str)
        budget_df['Budget'] = df[best_money_col]
        
        # Clean the data
        budget_df = self._clean_budget_data(budget_df)
        
        # Add table name prefix if provided
        if table_name:
            budget_df['Description'] = budget_df['Description'].apply(lambda x: f"{table_name}: {x}" if x and str(x) != 'nan' else table_name)
        
        return budget_df
    
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
            
            # Check if column name contains money keywords
            if any(keyword in col_str for keyword in MONEY_KEYWORDS):
                money_columns.append(col)
                continue
            
            # Check if column values contain monetary patterns
            has_money_pattern = False
            non_null_values = df[col].dropna().astype(str)
            
            for value in non_null_values.head(20):  # Check first 20 non-null values
                if MONEY_PATTERN.search(str(value)):
                    has_money_pattern = True
                    break
            
            if has_money_pattern:
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
        
        # Try to extract numbers from string with currency symbols
        # Remove currency symbols and extract numbers
        numeric_match = re.search(r'[\d,]+\.?\d*', value_str.replace(',', ''))
        
        if numeric_match:
            try:
                return float(numeric_match.group())
            except (ValueError, TypeError):
                pass
        
        return None
    
    def _select_best_budget_columns(self, combined_budget):
        """
        Select the best budget data if multiple tables were processed.
        
        Args:
            combined_budget: Combined DataFrame from multiple tables
            
        Returns:
            pandas.DataFrame: Best budget data limited to reasonable number of rows
        """
        if combined_budget.empty:
            return combined_budget
        
        # If too many rows, select the most relevant ones
        if len(combined_budget) > 50:
            # Sort by budget amount (descending) and take top 50
            combined_budget = combined_budget.sort_values('Budget', ascending=False).head(50)
        
        return combined_budget.reset_index(drop=True)


def get_budget_extractor():
    """Factory function to get a BudgetExtractor instance."""
    return BudgetExtractor()
