#!/usr/bin/env python3
"""
Debug script to isolate the pandas Series boolean error.
"""
import traceback
from main import TableExtractionOrchestrator

def test_pdf():
    """Test processing a single PDF file with error tracing."""
    try:
        orchestrator = TableExtractionOrchestrator()
        print("✅ Orchestrator initialized successfully")
        
        print("🔄 Downloading file from blob storage...")
        file_bytes = orchestrator.blob_manager.download_file("Appraisal-7c7215a75fbd4727a8160670c113e4e6.pdf")
        print(f"✅ File downloaded: {len(file_bytes)} bytes")
        
        print("🔄 Processing file with OCR...")
        tables = orchestrator.pdf_image_processor.process_file(file_bytes, '.pdf')
        print(f"✅ OCR completed, found {len(tables)} tables")
        
        print("🔄 Getting table metadata...")
        metadata = orchestrator.pdf_image_processor.get_table_metadata(tables)
        print("✅ Metadata generated")
        
        print("🔄 Filtering for budget tables...")
        budget_tables = orchestrator.pdf_image_processor.filter_budget_tables(tables)
        print(f"✅ Found {len(budget_tables)} budget tables")
        
        print("🔄 Extracting budget data...")
        budget_data = orchestrator.budget_extractor.extract_budget_from_tables(tables)
        print(f"✅ Budget extraction completed")
        
        print("✅ All processing completed successfully!")
        
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    test_pdf()
