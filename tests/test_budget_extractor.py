"""
Test cases for Budget Extractor - the most critical component for extracting budget data.
Tests focus on the core functionality that was recently fixed for pandas Series issues.
"""
import unittest
import pandas as pd
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from budget_extractor import BudgetExtractor


class TestBudgetExtractor(unittest.TestCase):
    """Test cases for BudgetExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = BudgetExtractor()
    
    def test_extract_budget_from_simple_table(self):
        """Test extraction from a simple table with clear budget data."""
        df = pd.DataFrame({
            'Description': ['Labor', 'Materials', 'Equipment'],
            'Amount': ['$1,000', '$500', '$200']
        })
        
        result = self.extractor._extract_budget_from_single_table(df)
        
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 3)
        self.assertIn('Description', result.columns)
        self.assertIn('Budget', result.columns)
        self.assertEqual(result.iloc[0]['Description'], 'Labor')
        self.assertEqual(result.iloc[0]['Budget'], 1000.0)
    
    def test_extract_budget_with_totals(self):
        """Test extraction that includes total rows."""
        df = pd.DataFrame({
            'Item': ['Service A', 'Service B', 'Total Cost'],
            'Cost': [100, 200, 300]
        })
        
        result = self.extractor._extract_budget_from_single_table(df)
        
        self.assertFalse(result.empty)
        # Should include the total row
        descriptions = result['Description'].tolist()
        self.assertTrue(any('Total' in str(desc) for desc in descriptions))
    
    def test_extract_budget_empty_table(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        
        result = self.extractor._extract_budget_from_single_table(df)
        
        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), ['Description', 'Budget'])
    
    def test_identify_money_columns(self):
        """Test money column identification."""
        df = pd.DataFrame({
            'Description': ['Item 1', 'Item 2'],
            'Price': ['$100', '$200'],
            'Quantity': [1, 2],
            'Total': ['$100', '$400'],
            'Notes': ['Note 1', 'Note 2']
        })
        
        money_cols = self.extractor._identify_money_columns(df)
        
        # Should identify Price and Total as money columns
        self.assertIn('Price', money_cols)
        self.assertIn('Total', money_cols)
        self.assertNotIn('Description', money_cols)
        self.assertNotIn('Notes', money_cols)
    
    def test_identify_label_columns(self):
        """Test label column identification."""
        df = pd.DataFrame({
            'Description': ['Item 1', 'Item 2'],
            'Name': ['Service A', 'Service B'],
            'Amount': [100, 200],
            'ID': [1, 2]
        })
        
        label_cols = self.extractor._identify_label_columns(df)
        
        # Should identify Description and Name as label columns
        self.assertIn('Description', label_cols)
        self.assertIn('Name', label_cols)
        self.assertNotIn('Amount', label_cols)
    
    def test_extract_numeric_value(self):
        """Test numeric value extraction from various formats."""
        test_cases = [
            ('$1,234.56', 1234.56),
            ('€2.500,75', 2500.75),  # European format
            ('(500)', -500),  # Negative in parentheses
            ('₹1,00,000', 100000),  # Indian format
            ('1.5K', 1.5),  # Should extract base number
            ('', None),
            ('Not a number', None),
            ('$0', 0),
            ('100.50 USD', 100.50)
        ]
        
        for input_val, expected in test_cases:
            with self.subTest(input_val=input_val):
                result = self.extractor._extract_numeric_value(input_val)
                if expected is None:
                    self.assertIsNone(result)
                else:
                    self.assertAlmostEqual(result, expected, places=2)
    
    def test_clean_budget_data(self):
        """Test budget data cleaning functionality."""
        df = pd.DataFrame({
            'Description': ['Valid Item', '', 'Another Item', 'nan'],
            'Budget': [100, 200, None, 'invalid']
        })
        df['_is_total'] = False
        
        # Test the cleaning method
        cleaned = self.extractor._clean_budget_data_with_totals(df)
        
        # Should remove empty descriptions and invalid budget values
        self.assertTrue(len(cleaned) <= 2)  # At most 2 valid rows
        self.assertTrue(all(str(desc).strip() != '' for desc in cleaned['Description']))
        self.assertTrue(all(pd.notna(budget) for budget in cleaned['Budget']))
    
    def test_extract_from_multiple_tables(self):
        """Test extraction from multiple tables (list input)."""
        table1 = pd.DataFrame({
            'Description': ['Item 1', 'Item 2'],
            'Cost': [100, 200]
        })
        table2 = pd.DataFrame({
            'Service': ['Service A', 'Service B'],
            'Price': ['$300', '$400']
        })
        
        tables = [table1, table2]
        result = self.extractor.extract_budget_from_tables(tables)
        
        self.assertFalse(result.empty)
        self.assertIn('Description', result.columns)
        self.assertIn('Budget', result.columns)
    
    def test_extract_from_table_with_sheet_names(self):
        """Test extraction from tables with sheet name tuples."""
        table_data = [
            ('Sheet1', pd.DataFrame({
                'Description': ['Labor'],
                'Amount': [1000]
            })),
            ('Sheet2', pd.DataFrame({
                'Item': ['Materials'],
                'Cost': [500]
            }))
        ]
        
        result = self.extractor.extract_budget_from_tables(table_data)
        
        self.assertFalse(result.empty)
        # Check that sheet names are included in descriptions
        descriptions = result['Description'].tolist()
        self.assertTrue(any('Sheet1' in str(desc) for desc in descriptions))
    
    def test_multiindex_column_handling(self):
        """Test handling of MultiIndex columns (the fix for pandas Series error)."""
        # Create a DataFrame that might have MultiIndex-like structure
        df = pd.DataFrame({
            ('', ''): ['Item 1', 'Item 2', 'Item 3'],  # Empty string columns
            ('Cost', 'USD'): [100, 200, 300],
            ('Notes', ''): ['Note 1', 'Note 2', 'Note 3']
        })
        
        # This should not raise the pandas Series boolean error
        try:
            result = self.extractor._extract_budget_from_single_table(df)
            # Test passes if no exception is raised
            self.assertIsInstance(result, pd.DataFrame)
        except ValueError as e:
            if "truth value of a Series is ambiguous" in str(e):
                self.fail("Pandas Series boolean error not properly handled")
            else:
                raise
    
    def test_is_likely_money_value(self):
        """Test the money value detection helper."""
        test_cases = [
            ('1234.56', True),
            ('$100', True),
            ('100.00', True),
            ('0', True),
            ('-50.25', False),  # Negative values might be treated differently
            ('abc', False),
            ('', False),
            ('100K', True),
            ('1,234', True)
        ]
        
        for value, expected in test_cases:
            with self.subTest(value=value):
                result = self.extractor._is_likely_money_value(value)
                self.assertEqual(result, expected)


class TestBudgetExtractorEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = BudgetExtractor()
    
    def test_no_money_columns_found(self):
        """Test when no money columns are identified."""
        df = pd.DataFrame({
            'Name': ['John', 'Jane'],
            'Age': [25, 30],
            'City': ['NYC', 'LA']
        })
        
        result = self.extractor._extract_budget_from_single_table(df)
        
        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), ['Description', 'Budget'])
    
    def test_no_label_columns_found(self):
        """Test when no label columns are identified."""
        df = pd.DataFrame({
            'Amount': [100, 200, 300],
            'Cost': [50, 100, 150],
            'Price': [10, 20, 30]
        })
        
        result = self.extractor._extract_budget_from_single_table(df)
        
        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), ['Description', 'Budget'])
    
    def test_single_column_dataframe(self):
        """Test handling of single-column DataFrame."""
        df = pd.DataFrame({
            'Amount': ['$100', '$200', '$300']
        })
        
        result = self.extractor._extract_budget_from_single_table(df)
        
        self.assertTrue(result.empty)  # No label column available


if __name__ == '__main__':
    unittest.main()
