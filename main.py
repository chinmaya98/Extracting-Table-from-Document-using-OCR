"""
Main Table Extraction Orchestrator
This is the main entry point for the table extraction application.
Coordinates PDF/Image and Excel processing modules with Streamlit UI.
"""

import os
import sys
import argparse
import streamlit as st
from pdf_image_processor import get_pdf_image_processor
from excel_processor import get_excel_processor
from azure_config import get_blob_manager, get_azure_config
from budget_extractor import get_budget_extractor
from utils.currency_utils import contains_money
import ui_handler


class TableExtractionOrchestrator:
    """
    Main orchestrator that coordinates different file type processors.
    """
    
    def __init__(self):
        """Initialize the orchestrator with all necessary processors."""
        try:
            self.pdf_image_processor = get_pdf_image_processor()
            self.excel_processor = get_excel_processor()
            self.blob_manager = get_blob_manager()
            self.azure_config = get_azure_config()
            self.budget_extractor = get_budget_extractor()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize TableExtractionOrchestrator: {e}")
    
    def extract_tables_from_file(self, file_bytes, filename):
        """
        Extract tables from a file based on its extension.
        
        Args:
            file_bytes: File content as bytes
            filename: Original filename with extension
            
        Returns:
            Dictionary with extraction results
        """
        try:
            # Determine file extension
            file_extension = os.path.splitext(filename)[1].lower()
            
            # Route to appropriate processor
            if file_extension in ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                return self._process_pdf_image(file_bytes, file_extension, filename)
            elif file_extension in ['.xlsx', '.xls', '.csv']:
                return self._process_excel_csv(file_bytes, file_extension, filename)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to extract tables from '{filename}': {e}")
    
    def _process_pdf_image(self, file_bytes, file_extension, filename):
        """Process PDF or image file."""
        try:
            tables = self.pdf_image_processor.process_file(file_bytes, file_extension)
            metadata = self.pdf_image_processor.get_table_metadata(tables)
            
            # Filter for budget tables
            budget_tables = self.pdf_image_processor.filter_budget_tables(tables)
            
            # Extract budget data from all tables
            budget_data = self.budget_extractor.extract_budget_from_tables(tables)
            
            return {
                'success': True,
                'file_type': 'pdf_image',
                'filename': filename,
                'tables': tables,
                'budget_tables': budget_tables,
                'budget_data': budget_data,
                'metadata': metadata,
                'sheets': None,  # Not applicable for PDF/Image
                'processor_used': 'PDFImageProcessor'
            }
            
        except Exception as e:
            return {
                'success': False,
                'file_type': 'pdf_image',
                'filename': filename,
                'error': str(e),
                'tables': [],
                'budget_tables': [],
                'budget_data': None,
                'processor_used': 'PDFImageProcessor'
            }
    
    def _process_excel_csv(self, file_bytes, file_extension, filename):
        """Process Excel or CSV file."""
        try:
            tables_list, sheets_dict, metadata = self.excel_processor.process_file(
                file_bytes, file_extension, filename
            )
            
            # Filter for budget tables
            budget_tables_list = self.excel_processor.filter_budget_tables(tables_list)
            
            # Extract just the DataFrames for consistency with PDF/Image output
            tables = [df for sheet_name, df in tables_list]
            budget_tables = [df for sheet_name, df in budget_tables_list]
            
            # Extract budget data from all tables (using the list with sheet names)
            budget_data = self.budget_extractor.extract_budget_from_tables(tables_list)
            
            return {
                'success': True,
                'file_type': 'excel_csv',
                'filename': filename,
                'tables': tables,
                'budget_tables': budget_tables,
                'budget_data': budget_data,
                'metadata': metadata,
                'sheets': sheets_dict,
                'tables_with_names': tables_list,  # Include sheet names
                'budget_tables_with_names': budget_tables_list,
                'processor_used': 'ExcelProcessor'
            }
            
        except Exception as e:
            return {
                'success': False,
                'file_type': 'excel_csv',
                'filename': filename,
                'error': str(e),
                'tables': [],
                'budget_tables': [],
                'budget_data': None,
                'sheets': {},
                'processor_used': 'ExcelProcessor'
            }
    
    def extract_tables_from_blob(self, blob_name):
        """
        Extract tables from a file stored in Azure Blob Storage.
        
        Args:
            blob_name: Name of the blob to process
            
        Returns:
            Dictionary with extraction results
        """
        try:
            # Download file from blob storage
            file_bytes = self.blob_manager.download_file(blob_name)
            
            # Extract tables
            result = self.extract_tables_from_file(file_bytes, blob_name)
            
            # Add blob information to result
            result['source'] = 'blob_storage'
            result['blob_name'] = blob_name
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'filename': blob_name,
                'source': 'blob_storage',
                'error': str(e),
                'tables': [],
                'budget_tables': [],
                'budget_data': None
            }
    
    def list_available_files(self, file_types=None):
        """
        List available files in blob storage.
        
        Args:
            file_types: List of file extensions to filter by (e.g., ['.pdf', '.xlsx'])
            
        Returns:
            List of available files with metadata
        """
        try:
            if file_types is None:
                file_types = ['.pdf', '.xlsx', '.xls', '.csv', '.jpg', '.jpeg', '.png', '.tiff']
            
            # Convert to tuple for blob manager
            extensions = tuple(file_types)
            
            return self.blob_manager.list_files_with_info(extensions)
            
        except Exception as e:
            raise RuntimeError(f"Failed to list available files: {e}")
    
    def batch_process_files(self, blob_names=None, file_types=None):
        """
        Process multiple files in batch.
        
        Args:
            blob_names: List of specific blob names to process (optional)
            file_types: List of file extensions to process (optional)
            
        Returns:
            List of processing results
        """
        try:
            if blob_names is None:
                # Get list of files to process
                files_info = self.list_available_files(file_types)
                blob_names = [file_info['name'] for file_info in files_info]
            
            results = []
            for blob_name in blob_names:
                try:
                    result = self.extract_tables_from_blob(blob_name)
                    results.append(result)
                except Exception as e:
                    results.append({
                        'success': False,
                        'filename': blob_name,
                        'error': str(e),
                        'tables': [],
                        'budget_tables': [],
                        'budget_data': None
                    })
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Failed to batch process files: {e}")
    
    def get_processing_summary(self, results):
        """
        Generate a summary of processing results.
        
        Args:
            results: List of processing results
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_files': len(results),
            'successful_files': 0,
            'failed_files': 0,
            'total_tables': 0,
            'total_budget_tables': 0,
            'total_budget_items': 0,
            'files_with_budget_data': 0,
            'file_types': {},
            'errors': []
        }
        
        for result in results:
            if result['success']:
                summary['successful_files'] += 1
                summary['total_tables'] += len(result.get('tables', []))
                summary['total_budget_tables'] += len(result.get('budget_tables', []))
                
                # Count budget data items
                budget_data = result.get('budget_data')
                if budget_data is not None and not budget_data.empty:
                    summary['total_budget_items'] += len(budget_data)
                    summary['files_with_budget_data'] += 1
                
                # Count file types
                file_type = result.get('file_type', 'unknown')
                summary['file_types'][file_type] = summary['file_types'].get(file_type, 0) + 1
            else:
                summary['failed_files'] += 1
                summary['errors'].append({
                    'filename': result.get('filename', 'unknown'),
                    'error': result.get('error', 'Unknown error')
                })
        
        return summary


def run_streamlit_app():
    """Run the Streamlit web application."""
    try:
        # Initialize orchestrator
        orchestrator = TableExtractionOrchestrator()
        
        # Initialize Streamlit UI
        ui_handler.initialize_app()
        
        # Handle blob interaction through UI
        selected_blob_file = ui_handler.handle_blob_interaction(orchestrator.blob_manager)
        
        if selected_blob_file:
            # Extract and display tables through UI
            result = orchestrator.extract_tables_from_blob(selected_blob_file)
            ui_handler.display_extraction_results(result)
            
    except Exception as e:
        st.error(f"Application error: {e}")


def run_cli_mode(args):
    """Run in command-line mode."""
    try:
        orchestrator = TableExtractionOrchestrator()
        
        if args.list_files:
            # List available files
            files = orchestrator.list_available_files()
            print(f"Found {len(files)} files in blob storage:")
            for file_info in files:
                print(f"  - {file_info['name']} ({file_info['size']} bytes)")
        
        elif args.process_file:
            # Process single file
            result = orchestrator.extract_tables_from_blob(args.process_file)
            if result['success']:
                print(f"✅ Successfully processed '{args.process_file}'")
                print(f"   Tables found: {len(result['tables'])}")
                print(f"   Budget tables: {len(result['budget_tables'])}")
                
                # Display budget extraction results
                budget_data = result.get('budget_data')
                if budget_data is not None and not budget_data.empty:
                    print(f"   Budget items extracted: {len(budget_data)}")
                    print("\n📊 Budget Data:")
                    print(budget_data.to_string(index=False, max_rows=10))
                    if len(budget_data) > 10:
                        print(f"   ... and {len(budget_data) - 10} more items")
                else:
                    print("   No budget data found")
            else:
                print(f"❌ Failed to process '{args.process_file}': {result['error']}")
        
        elif args.batch_process:
            # Batch process files
            file_types = args.file_types.split(',') if args.file_types else None
            results = orchestrator.batch_process_files(file_types=file_types)
            summary = orchestrator.get_processing_summary(results)
            
            print(f"📊 Batch Processing Summary:")
            print(f"   Total files: {summary['total_files']}")
            print(f"   Successful: {summary['successful_files']}")
            print(f"   Failed: {summary['failed_files']}")
            print(f"   Total tables: {summary['total_tables']}")
            print(f"   Budget tables: {summary['total_budget_tables']}")
            print(f"   Budget items extracted: {summary['total_budget_items']}")
            print(f"   Files with budget data: {summary['files_with_budget_data']}")
            
            # Show detailed budget results if requested
            if args.show_budget_details:
                print("\n💰 Budget Details by File:")
                for result in results:
                    if result['success']:
                        budget_data = result.get('budget_data')
                        if budget_data is not None and not budget_data.empty:
                            print(f"\n📄 {result['filename']}:")
                            print(budget_data.to_string(index=False, max_rows=5))
                            if len(budget_data) > 5:
                                print(f"     ... and {len(budget_data) - 5} more items")
                        else:
                            print(f"\n📄 {result['filename']}: No budget data found")
            
            if summary['errors']:
                print("\n❌ Errors:")
                for error in summary['errors']:
                    print(f"   - {error['filename']}: {error['error']}")
        
        else:
            print("No action specified. Use --help for available options.")
            
    except Exception as e:
        print(f"❌ CLI error: {e}")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Table Extraction Application")
    parser.add_argument('--mode', choices=['web', 'cli'], default='web',
                       help='Run mode: web (Streamlit) or cli (command line)')
    parser.add_argument('--list-files', action='store_true',
                       help='List available files in blob storage (CLI mode)')
    parser.add_argument('--process-file', type=str,
                       help='Process a specific file from blob storage (CLI mode)')
    parser.add_argument('--batch-process', action='store_true',
                       help='Batch process all files (CLI mode)')
    parser.add_argument('--show-budget-details', action='store_true',
                       help='Show detailed budget extraction results (CLI mode)')
    parser.add_argument('--file-types', type=str,
                       help='Comma-separated file extensions to filter (e.g., .pdf,.xlsx)')
    
    args = parser.parse_args()
    
    if args.mode == 'web':
        # Run Streamlit app
        run_streamlit_app()
    else:
        # Run in CLI mode
        run_cli_mode(args)


if __name__ == "__main__":
    main()
