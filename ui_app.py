"""
Trinity Online - Budget Table Extractor UI
Clean, professional interface for the Trinity workflow.
"""
import streamlit as st
import pandas as pd
import json
from io import StringIO, BytesIO
import os
from dotenv import load_dotenv

# Import our extractors
from table_extractor_pdf_image import PDFImageTableExtractor, AzureConfig
from table_extractor_excel import ExcelTableExtractor
from ai_budget_extractor import AIBudgetExtractor
from column_selector import apply_column_selection
from blob_file_manager import get_blob_file_manager

# Load environment variables
load_dotenv()


class TrinityUIApp:
    """Main Trinity Online UI Application."""
    
    def __init__(self):
        """Initialize the Trinity UI application."""
        # Initialize Azure configuration
        self.azure_config = AzureConfig()
        
        # Initialize services with proper dependencies
        document_client = self.azure_config.get_document_client()
        self.pdf_image_extractor = PDFImageTableExtractor(document_client)
        self.excel_extractor = ExcelTableExtractor()
        self.blob_file_manager = get_blob_file_manager()
        self.ai_budget_extractor = AIBudgetExtractor()
    
    def display_header(self):
        """Display the application header."""
        st.title("Trinity Online")
        st.subheader("Budget Table Extraction")
        st.markdown("---")
    

    
    def file_selection_section(self):
        """Handle file selection from blob storage using the new blob file manager."""
        st.header("File Selection")
        
        # Get file selection data from the new manager
        selection_data = self.blob_file_manager.get_file_selection_data()
        
        if selection_data['status'] == 'error':
            st.error("❌ Azure Blob Storage Configuration Issue")
            st.write(selection_data['message'])
            st.info("💡 Please check your .env file configuration:")
            st.code("AZURE_BLOB_CONNECTION_STRING=your_connection_string\nAZURE_BLOB_CONTAINER=your_container_name")
            return None
        
        # Display connection status
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Container", selection_data['container_name'])
        with col2:
            st.metric("Total Files", selection_data['total_files'])
        with col3:
            st.metric("Supported Files", selection_data['supported_count'])
        
        # Check if we have supported files
        if selection_data['supported_count'] == 0:
            st.warning("📂 No supported files found in blob storage")
            st.info("✅ Supported formats: Excel (.xlsx, .xls), PDF (.pdf), Images (.jpg, .png, etc.)")
            
            # Show file breakdown
            categories = selection_data['categories']
            if selection_data['total_files'] > 0:
                st.write("**Files by category:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f"📊 Excel: {len(categories['excel'])}")
                with col2:
                    st.write(f"📄 PDF: {len(categories['pdf'])}")
                with col3:
                    st.write(f"🖼️ Images: {len(categories['image'])}")
                with col4:
                    st.write(f"📁 Other: {len(categories['other'])}")
            
            return None
        
        # File selection interface
        st.write("**📋 Available Files for Processing:**")
        
        # Create file selection options
        file_options = []
        file_map = {}
        
        for file_info in selection_data['files']:
            display_name = f"{file_info['icon']} {file_info['name']} ({file_info['size_mb']} MB) - {file_info['category']}"
            file_options.append(display_name)
            file_map[display_name] = file_info
        
        # File selection dropdown
        selected_display_name = st.selectbox(
            "Choose a file to process:",
            file_options,
            index=0,
            help="Files are sorted by last modified date (newest first)"
        )
        
        if selected_display_name:
            selected_file_info = file_map[selected_display_name]
            
            # Show file details
            with st.expander("📄 File Details", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Name:** {selected_file_info['name']}")
                    st.write(f"**Category:** {selected_file_info['category']}")
                    st.write(f"**Size:** {selected_file_info['size_mb']} MB")
                with col2:
                    st.write(f"**Processor:** {selected_file_info['processor']}")
                    st.write(f"**Last Modified:** {selected_file_info['last_modified'].strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Content Type:** {selected_file_info['content_type']}")
            
            # Process button
            if st.button(f"🚀 Process {selected_file_info['category']} File", type="primary"):
                return self.process_file(selected_file_info['name'])
        
        return None
    
    def get_extractor_for_file(self, filename):
        """Get the appropriate extractor based on file type."""
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension in ['.xlsx', '.xls']:
            return self.excel_extractor
        else:
            return self.pdf_image_extractor
    
    def process_file(self, filename):
        """Process the selected file through the Trinity workflow."""
        if not filename:
            return None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Download file
            status_text.text("Step 1/5: Downloading file from blob storage...")
            progress_bar.progress(20)
            
            file_data = self.blob_file_manager.download_file(filename)
            if not file_data:
                st.error("Failed to download file from blob storage.")
                return None
            
            # Step 2: Extract tables using appropriate processor
            status_text.text("Step 2/5: Processing document with specialized AI...")
            progress_bar.progress(40)
            
            # Determine file extension and get appropriate extractor
            file_extension = os.path.splitext(filename)[1].lower()
            extractor = self.get_extractor_for_file(filename)
            
            try:
                # Process the file to extract tables using specialized extractor
                if file_extension in ['.xlsx', '.xls']:
                    # Excel files: use Excel extractor (returns tuple of tables and sheets)
                    extraction_result = extractor.process_file(file_data, file_extension, filename)
                    tables, individual_sheets = extraction_result
                else:
                    # PDF/Image files: use PDF/Image extractor (returns list of tables)
                    tables = extractor.process_file(file_data, file_extension)
                    individual_sheets = None
                
                # Step 3: Always generate AI Budget Summary first
                status_text.text("Step 3/5: Generating AI Budget Summary...")
                progress_bar.progress(50)
                
                # Collect all available content for AI analysis
                all_content = ""
                all_tables = []
                
                if tables:
                    # We have extracted tables
                    all_tables = tables
                    for i, table in enumerate(tables):
                        all_content += f"Table {i+1}:\n"
                        all_content += f"Headers: {list(table.columns)}\n"
                        all_content += table.head(5).to_string() + "\n\n"
                elif individual_sheets:
                    # We have Excel sheets but no budget tables
                    all_tables = list(individual_sheets.values())
                    for sheet_name, sheet_df in individual_sheets.items():
                        all_content += f"Sheet: {sheet_name}\n"
                        all_content += f"Headers: {list(sheet_df.columns)}\n" 
                        all_content += sheet_df.head(5).to_string() + "\n\n"
                else:
                    # No tables at all - still try to generate summary from file info
                    file_type = file_extension.upper().replace('.', '')
                    all_content = f"Document: {filename}\nFile Type: {file_type}\nNo structured tables detected."
                
                # Generate AI Budget Summary for ALL documents
                budget_summary = self.ai_budget_extractor.generate_budget_summary(
                    all_tables, all_content, filename
                )
                
                # Display AI Budget Summary prominently for every document
                st.success("🧠 **AI Budget Analysis**")
                st.info(budget_summary)
                st.markdown("---")
                
                # Now continue with table processing if tables exist
                if not tables:
                    status_text.text("Step 5/5: Analysis complete!")
                    progress_bar.progress(100)
                    
                    return {
                        'success': True,
                        'summary_only': True,
                        'summary': budget_summary,
                        'filename': filename,
                        'has_tables': False
                    }
                
                # For Excel files, we have individual sheets data
                if file_extension in ['.xlsx', '.xls'] and individual_sheets:
                    # Create enriched text context for AI processing
                    file_type = file_extension.upper().replace('.', '')
                    sheet_results = []
                    
                    # Process each sheet individually
                    for sheet_name, sheet_df in individual_sheets.items():
                        # Create context for this specific sheet
                        sheet_text = f"""
                        Document: {filename}
                        Sheet: {sheet_name}
                        File Type: {file_type}
                        Rows: {len(sheet_df)}
                        Columns: {list(sheet_df.columns)}
                        
                        This sheet contains structured data extracted for budget analysis.
                        """
                        
                        # Process with AI
                        ai_result = self.ai_budget_extractor.process_document_intelligence_data([sheet_df], sheet_text.strip())
                        
                        if ai_result.get('success', False):
                            processed_data = ai_result['extracted_data']
                            # Preserve original headers and make them bold (for display)
                            processed_data = self.ai_budget_extractor.format_budget_table_with_formatting(processed_data)
                        else:
                            # Use fallback processing for this sheet
                            processed_data = self.ai_budget_extractor._fallback_processing([sheet_df])
                        
                        if not processed_data.empty:
                            sheet_results.append({
                                'sheet_name': sheet_name,
                                'data': processed_data,
                                'original_headers': list(sheet_df.columns),
                                'rows': len(processed_data)
                            })
                    
                    if sheet_results:
                        result = {
                            'success': True,
                            'sheets': sheet_results,
                            'text': f"Excel file with {len(sheet_results)} processed sheets",
                            'is_multi_sheet': True
                        }
                    else:
                        # No budget tables found in Excel sheets, but summary already shown
                        status_text.text("Step 5/5: Processing complete!")
                        progress_bar.progress(100)
                        
                        return {
                            'success': True,
                            'summary_only': True,
                            'filename': filename,
                            'has_tables': True,
                            'reason': 'No budget tables in Excel sheets'
                        }
                else:
                    # Non-Excel files - use original processing
                    file_type = file_extension.upper().replace('.', '')
                    sheet_info = []
                    
                    # Collect source information
                    for i, table in enumerate(tables):
                        source = self.pdf_image_extractor.get_table_source_info(table)
                        sheet_info.append(f"Table {i+1}: {source} ({len(table)} rows)")
                    
                    text = f"""
                    Document: {filename}
                    File Type: {file_type}
                    Total Tables Found: {len(tables)}
                    
                    Table Sources:
                    {chr(10).join(sheet_info)}
                    
                    This document contains structured data extracted for budget analysis.
                    """
                    
                    result = {
                        'success': True,
                        'tables': tables,
                        'text': text.strip(),
                        'sheet_info': sheet_info,
                        'is_multi_sheet': False
                    }
                
            except Exception as e:
                # Even if processing fails, generate AI summary with whatever data we have
                status_text.text("Step 3/5: Generating AI Budget Summary...")
                progress_bar.progress(50)
                
                # Try to collect any available content for AI summary
                all_content = f"Document: {filename}\nFile processing encountered an error: {str(e)}\n"
                all_tables = []
                
                # Try to get basic file info for AI analysis
                try:
                    file_type = file_extension.upper().replace('.', '')
                    all_content += f"File Type: {file_type}\n"
                    
                    # If we have some tables from partial processing, use them
                    if 'tables' in locals() and tables:
                        all_tables = tables
                        all_content += f"Partial data extracted before error.\n"
                except:
                    pass
                
                # Generate AI Budget Summary even with limited data
                budget_summary = self.ai_budget_extractor.generate_budget_summary(
                    all_tables, all_content, filename
                )
                
                # Display the summary prominently
                st.success("🧠 **AI Budget Analysis**")
                st.info(budget_summary)
                st.markdown("---")
                
                # Show the error after displaying the summary
                st.warning(f"Processing encountered an issue: {str(e)}")
                
                status_text.text("Step 5/5: Analysis complete!")
                progress_bar.progress(100)
                
                return {
                    'success': True,
                    'summary_only': True,
                    'summary': budget_summary,
                    'filename': filename,
                    'error': str(e)
                }
            
            # Step 3: AI-enhanced budget extraction (if not already processed for Excel)
            if not result.get('is_multi_sheet', False):
                status_text.text("Step 3/5: AI analyzing budget data...")
                progress_bar.progress(60)
                
                tables = result.get('tables', [])
                text = result.get('text', '')
                
                if tables:
                    # Use AI to process the extracted data
                    ai_result = self.ai_budget_extractor.process_document_intelligence_data(tables, text)
                    
                    if ai_result.get('success', False):
                        budget_data = ai_result['extracted_data']
                        
                        # Step 4: Final formatting
                        status_text.text("Step 4/5: Formatting output...")
                        progress_bar.progress(80)
                        
                        # Apply final formatting
                        formatted_data = self.ai_budget_extractor.format_budget_table_with_formatting(budget_data)
                        
                        # Step 5: Complete
                        status_text.text("Step 5/5: Processing complete!")
                        progress_bar.progress(100)
                        
                        return {
                            'success': True,
                            'data': formatted_data,
                            'filename': filename,
                            'method': 'AI-enhanced',
                            'is_multi_sheet': False
                        }
                    else:
                        # Use AI extractor's built-in fallback processing
                        status_text.text("Step 3/5: Using fallback budget extraction...")
                        progress_bar.progress(60)
                        
                        # Use the AI extractor's fallback method
                        fallback_data = self.ai_budget_extractor._fallback_processing(tables)
                        
                        if not fallback_data.empty:
                            status_text.text("Step 5/5: Processing complete!")
                            progress_bar.progress(100)
                            
                            return {
                                'success': True,
                                'data': fallback_data,
                                'filename': filename,
                                'method': 'Fallback extraction',
                                'is_multi_sheet': False
                            }
            else:
                # Multi-sheet Excel already processed
                status_text.text("Step 5/5: Multi-sheet processing complete!")
                progress_bar.progress(100)
                
                return {
                    'success': True,
                    'sheets': result['sheets'],
                    'filename': filename,
                    'method': 'Sheet-by-sheet AI processing',
                    'is_multi_sheet': True
                }
            
            # Final fallback - generate AI summary when no budget data found
            status_text.text("Step 4/5: Generating AI budget summary...")
            progress_bar.progress(80)
            
            # Create text context from all extracted content
            file_type = file_extension.upper().replace('.', '')
            sheet_info = []
            
            # Collect source information and content
            all_text = f"Document: {filename}\nFile Type: {file_type}\n\n"
            for i, table in enumerate(tables):
                if hasattr(self, 'pdf_image_extractor'):
                    source = self.pdf_image_extractor.get_table_source_info(table)
                else:
                    source = f"Table {i+1}"
                sheet_info.append(f"Table {i+1}: {source} ({len(table)} rows)")
                
                # Add table content to text
                all_text += f"\nTable {i+1} ({source}):\n"
                all_text += f"Headers: {list(table.columns)}\n"
                all_text += table.head(5).to_string() + "\n\n"
            
            # Final fallback - processing complete but no budget tables, summary already shown
            status_text.text("Step 5/5: Analysis complete!")
            progress_bar.progress(100)
            
            return {
                'success': True,
                'summary_only': True,
                'filename': filename,
                'has_tables': True,
                'table_info': sheet_info,
                'reason': 'No budget tables found'
            }
            
        except Exception as e:
            # Final fallback - ensure AI summary is shown even in case of major errors
            try:
                status_text.text("Generating AI Budget Summary...")
                progress_bar.progress(90)
                
                # Generate summary with minimal information
                basic_content = f"Document: {filename}\nProcessing failed with error: {str(e)}"
                
                budget_summary = self.ai_budget_extractor.generate_budget_summary(
                    [], basic_content, filename
                )
                
                # Display the summary
                st.success("🧠 **AI Budget Analysis**") 
                st.info(budget_summary)
                st.markdown("---")
                st.warning(f"Processing encountered an issue: {str(e)}")
                
                return {
                    'success': True,
                    'summary_only': True,
                    'summary': budget_summary,
                    'filename': filename,
                    'error': str(e)
                }
            except:
                # Last resort fallback
                st.error(f"Processing failed: {str(e)}")
                return None
        
        finally:
            progress_bar.empty()
            status_text.empty()
    
    def display_results(self, result):
        """Display the processing results."""
        if not result:
            return
            
        # Handle summary-only results (AI analysis without table extraction)
        if result.get('summary_only', False):
            st.info("AI Budget Analysis completed successfully!")
            
            # Summary was already displayed during processing
            # Add any additional summary-specific information here if needed
            if 'error' in result:
                st.caption(f"Note: Processing had some limitations, but AI analysis was completed.")
            
            return
        
        # Handle normal table extraction results
        if not result.get('success', False):
            return
        
        filename = result['filename']
        method = result.get('method', 'AI-enhanced extraction')
        
        st.success(f"Budget data extracted successfully using {method}!")
        
        # Handle multi-sheet results
        if result.get('is_multi_sheet', False):
            sheets = result['sheets']
            
            # Display overall statistics
            col1, col2, col3 = st.columns(3)
            
            total_rows = sum(sheet['rows'] for sheet in sheets)
            
            with col1:
                st.metric("Sheets Processed", len(sheets))
            
            with col2:
                st.metric("Total Budget Items", total_rows)
            
            with col3:
                st.metric("Source File", filename.split('/')[-1] if '/' in filename else filename)
            
            st.markdown("---")
            
            # Display each sheet individually
            for i, sheet_result in enumerate(sheets):
                sheet_name = sheet_result['sheet_name']
                sheet_data = sheet_result['data']
                original_headers = sheet_result['original_headers']
                
                st.header(f"Sheet: {sheet_name}")
                
                # Sheet-specific statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Rows in Sheet", len(sheet_data))
                
                with col2:
                    # Count non-empty amount fields
                    if 'Amount' in sheet_data.columns:
                        amount_count = sheet_data['Amount'].notna().sum()
                        st.metric("Budget Items", amount_count)
                    else:
                        st.metric("Budget Items", "N/A")
                
                with col3:
                    st.metric("Original Columns", len(original_headers))
                
                # Show original headers info
                with st.expander(f"📋 Original Headers from {sheet_name}", expanded=False):
                    st.write("**Original table headers:**")
                    for header in original_headers:
                        st.write(f"• **{header}**")
                
                # Apply column selection to show only 3 best columns
                selected_data, selection_info = apply_column_selection(sheet_data, max_columns=3)
                
                # Display column selection information
                if selection_info['total_original'] > 3:
                    st.info(f"📊 **Display Optimization**: Showing {selection_info['total_selected']} most relevant columns out of {selection_info['total_original']} total columns")
                    
                    with st.expander(f"🔍 Column Selection Details", expanded=False):
                        st.write("**Selected columns for display:**")
                        for col in selection_info['selected_columns']:
                            st.write(f"• **{col}**")
                        
                        st.write("**All available columns:**")
                        for col in selection_info['original_columns']:
                            if col in selection_info['selected_columns']:
                                st.write(f"• ✅ **{col}** (displayed)")
                            else:
                                st.write(f"• ⏸️ {col} (hidden)")
                
                # Display the selected sheet data with bold headers
                st.write("**Top 3 Budget Columns:**")
                
                # Create a styled dataframe with bold headers
                styled_df = selected_data.style.set_table_styles([
                    {'selector': 'th', 'props': [('font-weight', 'bold'), ('background-color', '#f0f0f0')]}
                ])
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Individual sheet download options
                st.write("**Download Options for this Sheet:**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # CSV download for this sheet (selected columns)
                    csv_data = selected_data.to_csv(index=False)
                    st.download_button(
                        label=f"Download {sheet_name} as CSV (Top 3 Columns)",
                        data=csv_data,
                        file_name=f"budget_{sheet_name}_{filename.split('.')[0]}_selected.csv",
                        mime="text/csv",
                        key=f"csv_{i}_{sheet_name}"
                    )
                
                with col2:
                    # Excel download for this sheet (selected columns)
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        selected_data.to_excel(writer, index=False, sheet_name=sheet_name)
                        
                        # Make headers bold
                        workbook = writer.book
                        worksheet = workbook[sheet_name]
                        
                        from openpyxl.styles import Font
                        bold_font = Font(bold=True)
                        
                        # Apply bold formatting to header row
                        for cell in worksheet[1]:
                            cell.font = bold_font
                    
                    excel_data = excel_buffer.getvalue()
                    
                    st.download_button(
                        label=f"Download {sheet_name} as Excel (Top 3 Columns)",
                        data=excel_data,
                        file_name=f"budget_{sheet_name}_{filename.split('.')[0]}_selected.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"excel_{i}_{sheet_name}"
                    )
                
                with col3:
                    # JSON download for this sheet
                    json_data = sheet_data.to_json(orient='records', indent=2)
                    st.download_button(
                        label=f"Download {sheet_name} as JSON",
                        data=json_data,
                        file_name=f"budget_{sheet_name}_{filename.split('.')[0]}.json",
                        mime="application/json",
                        key=f"json_{i}_{sheet_name}"
                    )
                
                if i < len(sheets) - 1:  # Don't add separator after last sheet
                    st.markdown("---")
            
            # Combined download option
            st.header("📁 Combined Download Options")
            st.write("Download all processed sheets in a single file:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Combined CSV (all sheets)
                combined_csv = ""
                for sheet_result in sheets:
                    sheet_name = sheet_result['sheet_name']
                    sheet_data = sheet_result['data']
                    
                    combined_csv += f"# Sheet: {sheet_name}\n"
                    combined_csv += sheet_data.to_csv(index=False)
                    combined_csv += "\n\n"
                
                st.download_button(
                    label="Download All Sheets as CSV",
                    data=combined_csv,
                    file_name=f"budget_all_sheets_{filename.split('.')[0]}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Combined Excel (multiple sheets)
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    for sheet_result in sheets:
                        sheet_name = sheet_result['sheet_name']
                        sheet_data = sheet_result['data']
                        
                        # Write each sheet
                        sheet_data.to_excel(writer, index=False, sheet_name=f"Budget_{sheet_name}")
                        
                        # Make headers bold
                        workbook = writer.book
                        worksheet = workbook[f"Budget_{sheet_name}"]
                        
                        from openpyxl.styles import Font
                        bold_font = Font(bold=True)
                        
                        # Apply bold formatting to header row
                        for cell in worksheet[1]:
                            cell.font = bold_font
                
                excel_data = excel_buffer.getvalue()
                
                st.download_button(
                    label="Download All Sheets as Excel",
                    data=excel_data,
                    file_name=f"budget_all_sheets_{filename.split('.')[0]}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col3:
                # Combined JSON (all sheets)
                combined_json = {}
                for sheet_result in sheets:
                    sheet_name = sheet_result['sheet_name']
                    sheet_data = sheet_result['data']
                    combined_json[sheet_name] = sheet_data.to_dict('records')
                
                json_data = json.dumps(combined_json, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="Download All Sheets as JSON",
                    data=json_data,
                    file_name=f"budget_all_sheets_{filename.split('.')[0]}.json",
                    mime="application/json"
                )
        
        else:
            # Handle single table results (non-Excel or single sheet)
            data = result['data']
            
            # Display statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Rows", len(data))
            
            with col2:
                # Count non-empty amount fields
                if 'Amount' in data.columns:
                    amount_count = data['Amount'].notna().sum()
                    st.metric("Budget Items", amount_count)
                else:
                    st.metric("Budget Items", "N/A")
            
            with col3:
                st.metric("Source File", filename.split('/')[-1] if '/' in filename else filename)
            
            # Display the data table
            st.header("Extracted Budget Data")
            
            # Show source information if available
            if 'sheet_info' in result and result['sheet_info']:
                with st.expander("📋 Source Information", expanded=False):
                    st.write("**Tables processed from:**")
                    for info in result['sheet_info']:
                        st.write(f"• {info}")
            
            # Apply column selection to show only 3 best columns
            selected_data, selection_info = apply_column_selection(data, max_columns=3)
            
            # Display column selection information
            if selection_info['total_original'] > 3:
                st.info(f"📊 **Display Optimization**: Showing {selection_info['total_selected']} most relevant columns out of {selection_info['total_original']} total columns")
                
                with st.expander(f"🔍 Column Selection Details", expanded=False):
                    st.write("**Selected columns for display:**")
                    for col in selection_info['selected_columns']:
                        st.write(f"• **{col}**")
                    
                    st.write("**All available columns:**")
                    for col in selection_info['original_columns']:
                        if col in selection_info['selected_columns']:
                            st.write(f"• ✅ **{col}** (displayed)")
                        else:
                            st.write(f"• ⏸️ {col} (hidden)")
            
            # Display the selected data with bold headers  
            st.write("**Top 3 Budget Columns:**")
            
            # Create styled dataframe with bold headers
            styled_df = selected_data.style.set_table_styles([
                {'selector': 'th', 'props': [('font-weight', 'bold'), ('background-color', '#f0f0f0')]}
            ])
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Download options
            st.header("Export Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV download (selected columns)
                csv_data = selected_data.to_csv(index=False)
                st.download_button(
                    label="Download as CSV (Top 3 Columns)",
                    data=csv_data,
                    file_name=f"budget_data_{filename.split('.')[0]}_selected.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Excel download (selected columns)
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    selected_data.to_excel(writer, index=False, sheet_name='Budget Data')
                    
                    # Make headers bold
                    workbook = writer.book
                    worksheet = workbook['Budget Data']
                    
                    from openpyxl.styles import Font
                    bold_font = Font(bold=True)
                    
                    # Apply bold formatting to header row
                    for cell in worksheet[1]:
                        cell.font = bold_font
                
                excel_data = excel_buffer.getvalue()
                
                st.download_button(
                    label="Download as Excel (Top 3 Columns)",
                    data=excel_data,
                    file_name=f"budget_data_{filename.split('.')[0]}_selected.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col3:
                # JSON download (selected columns)
                json_data = selected_data.to_json(orient='records', indent=2)
                st.download_button(
                    label="Download as JSON (Top 3 Columns)",
                    data=json_data,
                    file_name=f"budget_data_{filename.split('.')[0]}_selected.json",
                    mime="application/json"
                )
    
    def run(self):
        """Run the main application."""
        st.set_page_config(
            page_title="Trinity Online",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # Custom CSS for clean styling
        st.markdown("""
        <style>
        .main > div {
            padding-top: 2rem;
        }
        .stButton > button {
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Display header
        self.display_header()
        
        # File processing section
        result = self.file_selection_section()
        
        # Display results if processing was successful
        if result:
            st.markdown("---")
            self.display_results(result)


def run_app():
    """Entry point for the application."""
    app = TrinityUIApp()
    app.run()


if __name__ == "__main__":
    run_app()
