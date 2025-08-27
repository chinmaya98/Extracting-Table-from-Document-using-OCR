"""
Test cases for currency utilities - the core money detection functionality.
These tests are critical as they determine which tables contain budget data.
"""
import unittest
import pandas as pd
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.currency_utils import contains_money, MONEY_PATTERN, MONEY_KEYWORDS


class TestCurrencyUtils(unittest.TestCase):
    """Test cases for currency detection utilities."""
    
    def test_contains_money_with_currency_symbols(self):
        """Test detection of currency symbols in DataFrame."""
        df = pd.DataFrame({
            'Description': ['Item 1', 'Item 2', 'Item 3'],
            'Amount': ['$100.50', '€200', '£300.75']
        })
        self.assertTrue(contains_money(df))
    
    def test_contains_money_with_column_headers(self):
        """Test detection based on money-related column headers."""
        df = pd.DataFrame({
            'Description': ['Item 1', 'Item 2'],
            'Budget': [100, 200],
            'Price': [50, 75]
        })
        self.assertTrue(contains_money(df))
    
    def test_contains_money_with_mixed_currency(self):
        """Test detection with various currency formats."""
        df = pd.DataFrame({
            'Item': ['Service A', 'Service B', 'Service C'],
            'Cost': ['₹5,000', '1,234.56 USD', '$0']
        })
        self.assertTrue(contains_money(df))
    
    def test_contains_money_no_monetary_data(self):
        """Test that non-monetary DataFrames return False."""
        df = pd.DataFrame({
            'Name': ['John', 'Jane', 'Bob'],
            'Age': [25, 30, 35],
            'City': ['NYC', 'LA', 'Chicago']
        })
        self.assertFalse(contains_money(df))
    
    def test_contains_money_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        self.assertFalse(contains_money(df))
    
    def test_contains_money_with_total_column(self):
        """Test detection of total/sum columns."""
        df = pd.DataFrame({
            'Description': ['Labor', 'Materials'],
            'Total': [1000, 500]
        })
        self.assertTrue(contains_money(df))
    
    def test_money_pattern_regex(self):
        """Test the money pattern regex directly."""
        test_cases = [
            ('$1,234.56', True),
            ('€100', True),
            ('£50.75', True),
            ('₹5,000', True),
            ('1234.56 USD', True),
            ('100 EUR', True),
            ('Regular text', False),
            ('123abc', False),
            ('', False)
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = bool(MONEY_PATTERN.search(text))
                self.assertEqual(result, expected, f"Pattern match failed for '{text}'")
    
    def test_money_keywords_coverage(self):
        """Test that essential money keywords are included."""
        essential_keywords = ['amount', 'budget', 'price', 'cost', 'total', 'value', 'usd', '$']
        for keyword in essential_keywords:
            self.assertIn(keyword, MONEY_KEYWORDS, f"Essential keyword '{keyword}' missing")
    
    def test_contains_money_with_nan_values(self):
        """Test handling of NaN values in monetary columns."""
        df = pd.DataFrame({
            'Description': ['Item 1', 'Item 2', None],
            'Amount': ['$100', None, '$300']
        })
        self.assertTrue(contains_money(df))


if __name__ == '__main__':
    unittest.main()
