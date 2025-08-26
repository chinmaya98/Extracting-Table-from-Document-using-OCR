"""
Budget Table Extractor Application
Simple Streamlit app that processes uploaded files and extracts budget information.
"""

import streamlit as st
import pandas as pd
import os
import tempfile
from main import TableExtractionOrchestrator
from budget_extractor import get_budget_extractor


def main():
    """Main application function."""
    # Configure Streamlit page
    st.set_page_config(
        page_title="Budget Table Extractor",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("💰 Budget Table Extractor")
    st.markdown("Upload a file (PDF, Image, Excel, or CSV) to extract budget/monetary information.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'jpg', 'jpeg', 'png', 'tiff', 'tif', 'xlsx', 'xls', 'csv'],
        help="Supported formats: PDF, Images (JPG, PNG, TIFF), Excel (XLSX, XLS), CSV"
    )
    
    if uploaded_file is not None:
        # Show file info
        st.info(f"Processing file: {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        # Process the file
        with st.spinner("Extracting tables and budget information..."):
            try:
                # Initialize processors
                orchestrator = TableExtractionOrchestrator()
                budget_extractor = get_budget_extractor()
                
                # Get file bytes
                file_bytes = uploaded_file.read()
                
                # Extract tables using the orchestrator
                result = orchestrator.extract_tables_from_file(file_bytes, uploaded_file.name)
                
                if result and 'tables' in result:
                    tables = result['tables']
                    
                    if tables:
                        # Extract budget information
                        budget_df = budget_extractor.extract_budget_from_tables(tables)
                        
                        if not budget_df.empty:
                            st.success(f"✅ Found {len(budget_df)} budget entries!")
                            
                            # Display the budget table
                            st.subheader("📊 Budget Table")
                            st.dataframe(budget_df, use_container_width=True)
                            
                            # Provide download option
                            csv_data = budget_df.to_csv(index=False)
                            st.download_button(
                                label="⬇️ Download Budget Table as CSV",
                                data=csv_data,
                                file_name=f"budget_{uploaded_file.name.split('.')[0]}.csv",
                                mime="text/csv",
                                key="budget_download"
                            )
                            
                            # Show some statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Entries", len(budget_df))
                            with col2:
                                if budget_df['Budget'].dtype in ['float64', 'int64']:
                                    total_budget = budget_df['Budget'].sum()
                                    st.metric("Total Budget", f"${total_budget:,.2f}")
                            with col3:
                                if budget_df['Budget'].dtype in ['float64', 'int64']:
                                    avg_budget = budget_df['Budget'].mean()
                                    st.metric("Average Amount", f"${avg_budget:,.2f}")
                        
                        else:
                            st.warning("⚠️ No budget/monetary information found in the uploaded file.")
                            st.info("The file was processed, but no columns containing budget, amounts, or monetary values were detected.")
                    
                    else:
                        st.warning("⚠️ No tables found in the uploaded file.")
                
                else:
                    st.error("❌ Failed to process the file. Please check the file format and try again.")
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("Please ensure the file is not corrupted and try again.")


if __name__ == "__main__":
    main()
