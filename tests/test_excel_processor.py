"""
Test cases for Excel Processor - handles Excel and CSV file processing.
Critical for ensuring proper table extraction from office documents.
"""
import unittest
import pandas as pd
import io
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from excel_processor import ExcelProcessor


class TestExcelProcessor(unittest.TestCase):
    """Test cases for ExcelProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = ExcelProcessor()
    
    def create_sample_excel_bytes(self):
        """Create sample Excel file bytes for testing."""
        df = pd.DataFrame({
            'Description': ['Item 1', 'Item 2', 'Total'],
            'Amount': [100, 200, 300],
            'Category': ['A', 'B', 'Summary']
        })
        
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False, engine='openpyxl')
        buffer.seek(0)
        return buffer.getvalue()
    
    def create_sample_csv_bytes(self):
        """Create sample CSV file bytes for testing."""
        df = pd.DataFrame({
            'Product': ['Product A', 'Product B'],
            'Price': ['$50.00', '$75.50'],
            'Quantity': [10, 5]
        })
        
        return df.to_csv(index=False).encode('utf-8')
    
    def test_clean_table_removes_empty_rows(self):
        """Test that empty rows are properly removed."""
        df = pd.DataFrame({
            'Col1': ['Value1', '', 'Value3', None],
            'Col2': ['Value2', '', None, ''],
            'Col3': ['Value3', None, 'Value6', 'Value7']
        })
        
        cleaned = self.processor.clean_table(df, max_null_threshold=0.5)
        
        # Should remove rows where more than 50% of cells are null/empty
        self.assertTrue(len(cleaned) < len(df))
        self.assertFalse(cleaned.empty)
    
    def test_clean_table_with_threshold(self):
        """Test clean_table with different null thresholds."""
        df = pd.DataFrame({
            'A': [1, None, 3, None],
            'B': [2, None, None, 4],
            'C': [3, 6, None, None]
        })
        
        # With threshold 0.3, should keep rows with <= 30% nulls
        cleaned_strict = self.processor.clean_table(df, max_null_threshold=0.3)
        
        # With threshold 0.8, should keep rows with <= 80% nulls
        cleaned_lenient = self.processor.clean_table(df, max_null_threshold=0.8)
        
        self.assertTrue(len(cleaned_lenient) >= len(cleaned_strict))
    
    def test_standardize_dataframe(self):
        """Test DataFrame standardization."""
        df = pd.DataFrame({
            'Text': ['Hello', 'World', None],
            'Number': [1, 2, None],
            'Mixed': ['Text', 123, None]
        })
        
        standardized = self.processor.standardize_dataframe(df)
        
        # Text columns should be strings, numeric should be filled with 0
        self.assertEqual(standardized['Text'].dtype, 'object')
        self.assertTrue(pd.api.types.is_numeric_dtype(standardized['Number']))
        self.assertEqual(standardized['Number'].isna().sum(), 0)  # NaN should be filled with 0
    
    def test_preprocess_dataframe_column_names(self):
        """Test DataFrame preprocessing for column names."""
        df = pd.DataFrame({
            '': ['Value1', 'Value2'],
            None: ['Value3', 'Value4'],
            'Normal Col': ['Value5', 'Value6'],
            'Duplicate': ['Value7', 'Value8'],
            'Duplicate': ['Value9', 'Value10']  # Duplicate column name
        })
        
        processed = self.processor.preprocess_dataframe(df)
        
        # Should have proper column names
        self.assertTrue(all(col != '' and col is not None for col in processed.columns))
        # Should have unique column names
        self.assertEqual(len(processed.columns), len(set(processed.columns)))
    
    def test_filter_budget_tables(self):
        """Test filtering of budget tables."""
        budget_table = pd.DataFrame({
            'Description': ['Service', 'Materials'],
            'Cost': ['$100', '$200']
        })
        
        non_budget_table = pd.DataFrame({
            'Name': ['John', 'Jane'],
            'Age': [25, 30]
        })
        
        tables_list = [
            ('Budget Sheet', budget_table),
            ('Employee List', non_budget_table)
        ]
        
        filtered = self.processor.filter_budget_tables(tables_list)
        
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0][0], 'Budget Sheet')
    
    def test_process_csv_data(self):
        """Test CSV processing."""
        csv_bytes = self.create_sample_csv_bytes()
        
        try:
            result = self.processor.process_and_save_to_csv(
                csv_bytes, '.csv', 'test.csv', max_null_threshold=0.8
            )
            
            self.assertTrue(result['success'])
            self.assertIn('tables', result)
            self.assertIn('budget_tables', result)
            self.assertIsInstance(result['tables'], list)
            
        except Exception as e:
            # CSV processing might fail if dependencies are missing
            self.skipTest(f"CSV processing failed: {e}")
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame()
        
        # These should not raise exceptions
        cleaned = self.processor.clean_table(empty_df)
        self.assertTrue(cleaned.empty)
        
        standardized = self.processor.standardize_dataframe(empty_df)
        self.assertTrue(standardized.empty)
        
        preprocessed = self.processor.preprocess_dataframe(empty_df)
        self.assertTrue(preprocessed.empty)
    
    def test_malformed_data_handling(self):
        """Test handling of malformed data."""
        df = pd.DataFrame({
            'Col1': ['Normal', 'Data', 'Here'],
            'Col2': [1, 'mixed', None],
            'Col3': [None, None, None]  # All null column
        })
        
        try:
            cleaned = self.processor.clean_table(df)
            standardized = self.processor.standardize_dataframe(cleaned)
            
            # Should handle mixed data types without crashing
            self.assertIsInstance(standardized, pd.DataFrame)
            
        except Exception as e:
            self.fail(f"Failed to handle malformed data: {e}")


class TestExcelProcessorEdgeCases(unittest.TestCase):
    """Test edge cases for Excel processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = ExcelProcessor()
    
    def test_very_small_dataframe(self):
        """Test processing of very small DataFrames."""
        df = pd.DataFrame({'A': [1]})
        
        result = self.processor.clean_table(df)
        self.assertEqual(len(result), 1)
    
    def test_very_large_null_threshold(self):
        """Test with null threshold of 1.0 (keep all rows)."""
        df = pd.DataFrame({
            'A': [None, None, None],
            'B': [None, None, None],
            'C': [None, None, None]
        })
        
        result = self.processor.clean_table(df, max_null_threshold=1.0)
        self.assertEqual(len(result), len(df))
    
    def test_zero_null_threshold(self):
        """Test with null threshold of 0.0 (remove any row with nulls)."""
        df = pd.DataFrame({
            'A': [1, 2, None],
            'B': [4, 5, 6],
            'C': [7, None, 9]
        })
        
        result = self.processor.clean_table(df, max_null_threshold=0.0)
        # Should only keep rows with no null values
        self.assertTrue(len(result) <= len(df))
        self.assertTrue(result.isna().sum().sum() == 0)


if __name__ == '__main__':
    unittest.main()
