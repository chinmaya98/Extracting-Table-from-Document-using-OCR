import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import streamlit as st
from ui_handler import initialize_app, handle_blob_interaction, extract_and_display_tables, process_excel_sheets, download_table_as_json

class TestUIHandler(unittest.TestCase):

    @patch('ui_handler.st')
    def test_initialize_app(self, mock_st):
        """Test the initialize_app function."""
        initialize_app()
        mock_st.set_page_config.assert_called_once_with(page_title="Budget Table Extractor", layout="wide")
        mock_st.title.assert_called_once_with("📄 Budget Table Extractor")

    @patch('ui_handler.BlobManager')
    @patch('ui_handler.st')
    def test_handle_blob_interaction(self, mock_st, mock_blob_manager):
        """Test the handle_blob_interaction function."""
        mock_blob_manager.list_files.return_value = ['file1.csv', 'file2.xlsx']
        mock_st.selectbox.return_value = 'file1.csv'

        selected_file = handle_blob_interaction(mock_blob_manager)

        mock_blob_manager.list_files.assert_called_once()
        mock_st.selectbox.assert_called_once_with("Select a file from Blob Storage", ['file1.csv', 'file2.xlsx'])
        self.assertEqual(selected_file, 'file1.csv')

    @patch('ui_handler.st')
    @patch('ui_handler.TableExtractor')
    @patch('ui_handler.BlobManager')
    def test_extract_and_display_tables(self, mock_blob_manager, mock_extractor, mock_st):
        """Test the extract_and_display_tables function."""
        mock_blob_manager.download_file.return_value = b"mocked file bytes"
        mock_extractor.extract_from_pdf.return_value = [pd.DataFrame({'A': [1, 2], 'B': [3, 4]})]
        mock_extractor.filter_tables.return_value = [pd.DataFrame({'A': [1, 2], 'B': [3, 4]})]
        mock_extractor.clean_table.return_value = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})

        with patch('ui_handler.st.session_state', {"filtered_tables": []}):
            extract_and_display_tables(mock_blob_manager, mock_extractor, "mock_file.pdf")

        mock_blob_manager.download_file.assert_called_once_with("mock_file.pdf")
        mock_extractor.extract_from_pdf.assert_called_once_with(b"mocked file bytes")
        mock_extractor.filter_tables.assert_called_once()
        mock_extractor.clean_table.assert_called_once()
        mock_st.subheader.assert_called_with("📊 Table 1")

    @patch('ui_handler.st')
    @patch('ui_handler.read_csv_or_excel_file')
    def test_process_excel_sheets(self, mock_read_csv_or_excel_file, mock_st):
        """Test the process_excel_sheets function."""
        mock_read_csv_or_excel_file.return_value = ([], None, [], {
            'Sheet1': pd.DataFrame({'A': [1, 2], 'B': [3, 4]}),
            'Sheet2': pd.DataFrame({'C': [5, 6], 'D': [7, 8]})
        })

        with patch('ui_handler.st.session_state', {"filtered_tables": []}):
            process_excel_sheets(b"mocked file bytes", "mock_file.xlsx")

        mock_read_csv_or_excel_file.assert_called_once_with(b"mocked file bytes", "mock_file.xlsx")
        mock_st.subheader.assert_any_call("📊 Sheet: Sheet1")
        mock_st.subheader.assert_any_call("📊 Sheet: Sheet2")

    @patch('ui_handler.st')
    def test_download_table_as_json(self, mock_st):
        """Test the download_table_as_json function."""
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        download_table_as_json(df, 1)

        mock_st.download_button.assert_called_once_with(
            label="⬇️ Download Table 1 as JSON",
            data=df.to_json(orient="records", indent=2).encode("utf-8"),
            file_name="table_1.json",
            mime="application/json"
        )

if __name__ == '__main__':
    unittest.main()
