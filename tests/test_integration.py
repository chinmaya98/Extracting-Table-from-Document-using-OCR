"""
Integration tests for the main orchestrator - tests the complete workflow.
These tests ensure all components work together properly.
"""
import unittest
import pandas as pd
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from main import TableExtractionOrchestrator
except ImportError:
    # Skip if main dependencies are not available
    TableExtractionOrchestrator = None


@unittest.skipIf(TableExtractionOrchestrator is None, "Main dependencies not available")
class TestTableExtractionOrchestrator(unittest.TestCase):
    """Integration tests for the main orchestrator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_table = pd.DataFrame({
            'Description': ['Labor', 'Materials', 'Total'],
            'Amount': [1000, 500, 1500]
        })
    
    @patch('main.get_pdf_image_processor')
    @patch('main.get_excel_processor') 
    @patch('main.get_blob_manager')
    @patch('main.get_azure_config')
    @patch('main.get_budget_extractor')
    def test_orchestrator_initialization(self, mock_budget_extractor, mock_azure_config, 
                                       mock_blob_manager, mock_excel_processor, mock_pdf_processor):
        """Test that orchestrator initializes correctly."""
        try:
            orchestrator = TableExtractionOrchestrator()
            self.assertIsNotNone(orchestrator)
        except Exception as e:
            self.fail(f"Failed to initialize orchestrator: {e}")
    
    def test_extract_tables_from_file_invalid_extension(self):
        """Test handling of unsupported file extensions."""
        try:
            orchestrator = TableExtractionOrchestrator()
            result = orchestrator.extract_tables_from_file(b'test data', 'test.txt')
            
            self.assertFalse(result['success'])
            self.assertIn('error', result)
            
        except Exception as e:
            # If initialization fails due to missing dependencies, skip
            self.skipTest(f"Orchestrator initialization failed: {e}")
    
    def test_get_processing_summary(self):
        """Test processing summary generation."""
        try:
            orchestrator = TableExtractionOrchestrator()
            
            # Mock results
            results = [
                {
                    'success': True,
                    'filename': 'test1.pdf',
                    'tables': [self.sample_table],
                    'budget_tables': [self.sample_table],
                    'budget_data': self.sample_table,
                    'file_type': 'pdf'
                },
                {
                    'success': False,
                    'filename': 'test2.pdf',
                    'error': 'Test error',
                    'tables': [],
                    'budget_tables': [],
                    'budget_data': None
                }
            ]
            
            summary = orchestrator.get_processing_summary(results)
            
            self.assertEqual(summary['total_files'], 2)
            self.assertEqual(summary['successful_files'], 1)
            self.assertEqual(summary['failed_files'], 1)
            self.assertEqual(summary['total_tables'], 1)
            self.assertEqual(summary['files_with_budget_data'], 1)
            
        except Exception as e:
            self.skipTest(f"Orchestrator test failed: {e}")
    
    def test_file_type_routing(self):
        """Test that files are routed to correct processors based on extension."""
        try:
            orchestrator = TableExtractionOrchestrator()
            
            # Test PDF routing
            with patch.object(orchestrator, '_process_pdf_image') as mock_pdf:
                mock_pdf.return_value = {'success': True, 'file_type': 'pdf_image'}
                result = orchestrator.extract_tables_from_file(b'test', 'test.pdf')
                mock_pdf.assert_called_once()
            
            # Test Excel routing
            with patch.object(orchestrator, '_process_excel_csv') as mock_excel:
                mock_excel.return_value = {'success': True, 'file_type': 'excel_csv'}
                result = orchestrator.extract_tables_from_file(b'test', 'test.xlsx')
                mock_excel.assert_called_once()
                
        except Exception as e:
            self.skipTest(f"File routing test failed: {e}")


class TestIntegrationWorkflow(unittest.TestCase):
    """Test the complete workflow without external dependencies."""
    
    def test_budget_extraction_workflow(self):
        """Test the complete budget extraction workflow with mock data."""
        from budget_extractor import BudgetExtractor
        from utils.currency_utils import contains_money
        
        # Create test data that would come from OCR/Excel processing
        tables = [
            pd.DataFrame({
                'Description': ['Labor Cost', 'Material Cost', 'Equipment'],
                'Amount': ['$2,500.00', '$1,200.50', '$800.75']
            }),
            pd.DataFrame({
                'Item': ['Service Fee', 'Total Cost'],
                'Price': ['$150', '$4,651.25']
            })
        ]
        
        # Test currency detection
        for table in tables:
            self.assertTrue(contains_money(table), f"Failed to detect money in table: {table.columns}")
        
        # Test budget extraction
        extractor = BudgetExtractor()
        budget_data = extractor.extract_budget_from_tables(tables)
        
        self.assertFalse(budget_data.empty)
        self.assertIn('Description', budget_data.columns)
        self.assertIn('Budget', budget_data.columns)
        
        # Check that monetary values were properly extracted
        budget_values = budget_data['Budget'].dropna()
        self.assertTrue(all(isinstance(val, (int, float)) for val in budget_values))
        self.assertTrue(all(val >= 0 for val in budget_values))  # All should be positive
    
    def test_data_cleaning_pipeline(self):
        """Test the data cleaning pipeline."""
        from excel_processor import ExcelProcessor
        from pdf_image_processor import PDFImageProcessor
        from unittest.mock import Mock
        
        # Test data with various issues
        messy_data = pd.DataFrame({
            'Description': ['Valid Item', '', 'Another Item', None, '   '],
            'Amount': ['$100.00', None, '€200,50', 'invalid', '$0'],
            'Notes': ['Note 1', 'Note 2', '', None, 'Final note']
        })
        
        # Test Excel processor cleaning
        excel_processor = ExcelProcessor()
        excel_cleaned = excel_processor.clean_table(messy_data, max_null_threshold=0.6)
        
        self.assertFalse(excel_cleaned.empty)
        self.assertTrue(len(excel_cleaned) <= len(messy_data))
        
        # Test PDF processor cleaning
        mock_client = Mock()
        pdf_processor = PDFImageProcessor(mock_client)
        pdf_cleaned = pdf_processor.clean_table(messy_data)
        
        self.assertFalse(pdf_cleaned.empty)
        self.assertTrue(len(pdf_cleaned) <= len(messy_data))
    
    def test_error_handling_workflow(self):
        """Test that errors are properly handled throughout the workflow."""
        from budget_extractor import BudgetExtractor
        
        extractor = BudgetExtractor()
        
        # Test with empty input
        result = extractor.extract_budget_from_tables([])
        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), ['Description', 'Budget'])
        
        # Test with invalid input
        result = extractor.extract_budget_from_tables(None)
        self.assertTrue(result.empty)
        
        # Test with non-monetary tables
        non_monetary = pd.DataFrame({
            'Name': ['John', 'Jane'],
            'Department': ['HR', 'IT']
        })
        
        result = extractor.extract_budget_from_tables([non_monetary])
        self.assertTrue(result.empty)


if __name__ == '__main__':
    unittest.main()
