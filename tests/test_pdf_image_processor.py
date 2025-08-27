"""
Test cases for PDF Image Processor - handles OCR and table extraction from PDFs and images.
Critical for ensuring proper OCR functionality and table detection.
"""
import unittest
import pandas as pd
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf_image_processor import PDFImageProcessor


class TestPDFImageProcessor(unittest.TestCase):
    """Test cases for PDFImageProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = PDFImageProcessor()
    
    def test_clean_table_removes_empty_rows(self):
        """Test that the fixed clean_table method works properly."""
        df = pd.DataFrame({
            'Col1': ['Value1', '', 'Value3'],
            'Col2': ['Value2', '', ''],
            'Col3': ['Value3', '', 'Value6']
        })
        
        cleaned = self.processor.clean_table(df)
        
        # Should remove the row with all empty values (row index 1)
        self.assertTrue(len(cleaned) < len(df))
        self.assertFalse(cleaned.empty)
        # Verify no completely empty rows remain
        for _, row in cleaned.iterrows():
            row_has_content = any(str(val).strip() != '' for val in row)
            self.assertTrue(row_has_content)
    
    def test_clean_table_preserves_valid_rows(self):
        """Test that valid rows are preserved during cleaning."""
        df = pd.DataFrame({
            'Description': ['Item 1', 'Item 2', 'Item 3'],
            'Amount': [100, 200, 300],
            'Notes': ['Note 1', '', 'Note 3']
        })
        
        cleaned = self.processor.clean_table(df)
        
        # Should keep all rows as none are completely empty
        self.assertEqual(len(cleaned), len(df))
        self.assertTrue(all(cleaned['Description'] == df['Description']))
    
    def test_clean_table_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        
        cleaned = self.processor.clean_table(df)
        
        self.assertTrue(cleaned.empty)
    
    def test_clean_table_with_nan_values(self):
        """Test handling of NaN values."""
        df = pd.DataFrame({
            'A': [1, None, 3],
            'B': [None, None, None],  # All NaN column
            'C': ['Text', None, 'More text']
        })
        
        cleaned = self.processor.clean_table(df)
        
        # Should preserve rows with some valid content
        self.assertFalse(cleaned.empty)
        # Should handle NaN values properly
        self.assertEqual(cleaned.fillna("").isna().sum().sum(), 0)
    
    def test_filter_budget_tables(self):
        """Test filtering of budget tables."""
        budget_table = pd.DataFrame({
            'Item': ['Labor', 'Materials'],
            'Cost': ['$1000', '$500']
        })
        
        non_budget_table = pd.DataFrame({
            'Name': ['John', 'Jane'],
            'Department': ['IT', 'HR']
        })
        
        tables = [budget_table, non_budget_table]
        
        filtered = self.processor.filter_budget_tables(tables)
        
        # Should only include the budget table
        self.assertEqual(len(filtered), 1)
        self.assertTrue('Cost' in filtered[0].columns or 'Item' in filtered[0].columns)
    
    def test_get_table_metadata(self):
        """Test table metadata generation."""
        tables = [
            pd.DataFrame({
                'Description': ['Item 1', 'Item 2'],
                'Price': ['$100', '$200']
            }),
            pd.DataFrame({
                'Name': ['John', 'Jane'],
                'Age': [25, 30]
            })
        ]
        
        metadata = self.processor.get_table_metadata(tables)
        
        self.assertIsInstance(metadata, list)
        self.assertEqual(len(metadata), len(tables))
        
        for meta in metadata:
            self.assertIn('rows', meta)
            self.assertIn('columns', meta)
            self.assertIn('has_monetary_data', meta)
    
    def test_pandas_series_fix(self):
        """Test that the pandas Series boolean fix is working."""
        # Create a problematic DataFrame that would cause the original error
        df = pd.DataFrame({
            'Col1': ['  ', '', 'Valid'],  # Mix of whitespace and empty
            'Col2': ['', '  ', 'Data'],   # Mix of empty and whitespace
            'Col3': ['Text', '', '']      # Mix of valid and empty
        })
        
        try:
            # This should not raise the pandas Series boolean error
            cleaned = self.processor.clean_table(df)
            self.assertIsInstance(cleaned, pd.DataFrame)
            
            # Should properly handle mixed empty/whitespace content
            self.assertTrue(len(cleaned) > 0)  # Should keep rows with valid content
            
        except ValueError as e:
            if "truth value of a Series is ambiguous" in str(e):
                self.fail("Pandas Series boolean error not properly fixed")
            else:
                raise
    
    def test_clean_table_whitespace_handling(self):
        """Test proper handling of whitespace-only content."""
        df = pd.DataFrame({
            'A': ['Valid', '   ', 'Also Valid'],
            'B': ['', '   ', ''],
            'C': ['Text', '', 'More Text']
        })
        
        cleaned = self.processor.clean_table(df)
        
        # Row 1 (index 1) has only whitespace and empty strings, should be removed
        # Rows 0 and 2 have valid content, should be kept
        self.assertTrue(len(cleaned) <= len(df))
        
        # Check that no completely empty rows remain
        for _, row in cleaned.iterrows():
            has_content = any(str(val).strip() != '' for val in row)
            self.assertTrue(has_content, f"Found empty row: {row.tolist()}")
    
    def test_clean_table_mixed_data_types(self):
        """Test handling of mixed data types."""
        df = pd.DataFrame({
            'Numbers': [1, 2, None],
            'Text': ['Hello', '', 'World'],
            'Mixed': [1, 'text', None],
            'Booleans': [True, False, None]
        })
        
        try:
            cleaned = self.processor.clean_table(df)
            self.assertIsInstance(cleaned, pd.DataFrame)
            
            # Should handle different data types without errors
            self.assertTrue(len(cleaned) <= len(df))
            
        except Exception as e:
            self.fail(f"Failed to handle mixed data types: {e}")


class TestPDFImageProcessorEdgeCases(unittest.TestCase):
    """Test edge cases for PDF Image processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = PDFImageProcessor()
    
    def test_single_row_dataframe(self):
        """Test processing of single-row DataFrame."""
        df = pd.DataFrame({'A': ['Value'], 'B': ['Another Value']})
        
        cleaned = self.processor.clean_table(df)
        self.assertEqual(len(cleaned), 1)
    
    def test_single_column_dataframe(self):
        """Test processing of single-column DataFrame."""
        df = pd.DataFrame({'Only Column': ['Value1', '', 'Value3']})
        
        cleaned = self.processor.clean_table(df)
        
        # Should remove empty rows
        self.assertTrue(len(cleaned) < len(df))
        self.assertTrue(all(str(val).strip() != '' for val in cleaned['Only Column']))
    
    def test_all_empty_dataframe(self):
        """Test DataFrame with all empty/whitespace values."""
        df = pd.DataFrame({
            'A': ['', '   ', ''],
            'B': ['', '', '   '],
            'C': ['   ', '', '']
        })
        
        cleaned = self.processor.clean_table(df)
        
        # Should result in empty DataFrame as all rows are effectively empty
        self.assertTrue(cleaned.empty)
    
    def test_unicode_content(self):
        """Test handling of Unicode content."""
        df = pd.DataFrame({
            'Description': ['Item 1', 'Café', '日本語'],
            'Price': ['$100', '€50', '¥1000']
        })
        
        try:
            cleaned = self.processor.clean_table(df)
            self.assertEqual(len(cleaned), len(df))  # Should preserve all rows
            
        except Exception as e:
            self.fail(f"Failed to handle Unicode content: {e}")


if __name__ == '__main__':
    unittest.main()
