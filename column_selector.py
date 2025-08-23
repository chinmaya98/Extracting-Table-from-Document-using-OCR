"""
Column Selection Module for Trinity Online
This module handles the selection and display of specific columns from extracted tables.
"""
import pandas as pd
from typing import List, Dict, Optional, Tuple


class ColumnSelector:
    """Handle column selection and formatting for table display."""
    
    def __init__(self):
        """Initialize column selector with predefined column priorities."""
        # Priority order for common budget-related columns
        self.budget_column_priorities = [
            # Amount/Money columns (highest priority)
            'amount', 'cost', 'price', 'budget', 'total', 'value', 'expense', 'revenue',
            'balance', 'sum', 'subtotal', 'grand_total', 'net', 'gross',
            
            # Category/Description columns (second priority)
            'category', 'description', 'item', 'name', 'title', 'type', 'account',
            'department', 'project', 'service', 'product', 'line_item',
            
            # Date/Period columns (third priority)
            'date', 'month', 'year', 'quarter', 'period', 'fiscal_year',
            'start_date', 'end_date', 'due_date', 'created', 'updated',
            
            # Additional useful columns
            'quantity', 'unit', 'rate', 'hours', 'days', 'percent', 'percentage',
            'status', 'approved', 'pending', 'completed', 'notes', 'comments'
        ]
    
    def select_best_columns(self, df: pd.DataFrame, max_columns: int = 3) -> List[str]:
        """
        Select the best columns to display based on content and priority.
        
        Args:
            df: DataFrame to analyze
            max_columns: Maximum number of columns to select
            
        Returns:
            List of column names to display
        """
        if df.empty or len(df.columns) == 0:
            return []
        
        available_columns = list(df.columns)
        selected_columns = []
        
        # If we have fewer columns than requested, return all
        if len(available_columns) <= max_columns:
            return available_columns
        
        # Score each column based on priority and content
        column_scores = {}
        
        for col in available_columns:
            score = 0
            col_lower = str(col).lower().strip()
            
            # Check against priority keywords
            for i, keyword in enumerate(self.budget_column_priorities):
                if keyword in col_lower:
                    # Higher score for earlier (more important) keywords
                    score += (len(self.budget_column_priorities) - i) * 10
                    break
            
            # Bonus for columns with financial data
            if self._contains_financial_data(df[col]):
                score += 50
            
            # Bonus for non-empty columns
            non_empty_ratio = df[col].notna().sum() / len(df)
            score += non_empty_ratio * 20
            
            # Bonus for columns with varied content
            if df[col].nunique() > 1:
                score += 10
            
            column_scores[col] = score
        
        # Select top scoring columns
        sorted_columns = sorted(column_scores.items(), key=lambda x: x[1], reverse=True)
        selected_columns = [col for col, score in sorted_columns[:max_columns]]
        
        # Ensure we maintain original order when possible
        ordered_selection = [col for col in available_columns if col in selected_columns]
        
        return ordered_selection
    
    def _contains_financial_data(self, series: pd.Series) -> bool:
        """Check if a pandas Series contains financial/monetary data."""
        if series.empty:
            return False
        
        # Convert to string and check for financial indicators
        str_values = series.astype(str).str.lower()
        
        # Check for currency symbols
        currency_patterns = ['$', '€', '£', '¥', '₹', 'usd', 'eur', 'gbp']
        for pattern in currency_patterns:
            if str_values.str.contains(pattern, na=False).any():
                return True
        
        # Check for numeric values with decimal places (common in financial data)
        try:
            numeric_values = pd.to_numeric(series, errors='coerce')
            if numeric_values.notna().sum() > 0:
                # Check if values look like money (reasonable ranges, decimal places)
                non_null_values = numeric_values.dropna()
                if len(non_null_values) > 0:
                    # Look for decimal places or reasonable financial ranges
                    has_decimals = (non_null_values % 1 != 0).any()
                    reasonable_range = (non_null_values.abs() >= 0.01).any() and (non_null_values.abs() <= 1_000_000_000).any()
                    if has_decimals or reasonable_range:
                        return True
        except:
            pass
        
        return False
    
    def format_selected_columns(self, df: pd.DataFrame, selected_columns: List[str]) -> pd.DataFrame:
        """
        Format the selected columns for display.
        
        Args:
            df: Original DataFrame
            selected_columns: List of column names to include
            
        Returns:
            DataFrame with only selected columns, properly formatted
        """
        if df.empty or not selected_columns:
            return pd.DataFrame()
        
        # Filter to selected columns only
        available_columns = [col for col in selected_columns if col in df.columns]
        if not available_columns:
            return pd.DataFrame()
        
        result_df = df[available_columns].copy()
        
        # Clean up the data
        result_df = result_df.fillna("")
        
        # Store metadata about column selection (using proper pandas way)
        result_df.attrs['selected_columns'] = available_columns
        result_df.attrs['original_columns'] = list(df.columns)
        result_df.attrs['selection_info'] = f"Showing {len(available_columns)} of {len(df.columns)} columns"
        
        return result_df
    
    def get_column_selection_info(self, original_df: pd.DataFrame, selected_df: pd.DataFrame) -> Dict[str, any]:
        """
        Get information about the column selection process.
        
        Args:
            original_df: Original DataFrame
            selected_df: DataFrame after column selection
            
        Returns:
            Dictionary with selection information
        """
        return {
            'original_columns': list(original_df.columns),
            'selected_columns': selected_df.attrs.get('selected_columns', list(selected_df.columns)),
            'total_original': len(original_df.columns),
            'total_selected': len(selected_df.columns),
            'selection_summary': selected_df.attrs.get('selection_info', 'Column selection applied')
        }


def apply_column_selection(df: pd.DataFrame, max_columns: int = 3) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Apply intelligent column selection to a DataFrame.
    
    Args:
        df: Input DataFrame
        max_columns: Maximum number of columns to select
        
    Returns:
        Tuple of (selected_dataframe, selection_info)
    """
    selector = ColumnSelector()
    
    if df.empty:
        return df, {'original_columns': [], 'selected_columns': [], 'total_original': 0, 'total_selected': 0}
    
    # Select best columns
    selected_columns = selector.select_best_columns(df, max_columns)
    
    # Format the result
    result_df = selector.format_selected_columns(df, selected_columns)
    
    # Get selection info
    selection_info = selector.get_column_selection_info(df, result_df)
    
    return result_df, selection_info
