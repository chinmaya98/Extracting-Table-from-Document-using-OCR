# PDF Table & Budget Extractor with Azure Document Intelligence

This Streamlit app allows you to **extract tables and budget information** from PDF documents, Excel files, and images using Azure Document Intelligence OCR. The app provides both traditional table extraction and advanced budget/label extraction with standardized output.

---

## Features

### Traditional Table Extraction
- Extract tables from PDF files using Azure Document Intelligence
- Filter tables containing monetary values (amounts, prices, totals, etc.)
- View extracted tables in the browser
- Download results as CSV or JSON

### NEW: Budget & Label Extraction
- 🆕 **Extract budget data from PDFs, Excel files, and images**
- 🆕 **Standardized 3-column output**: Label1, Label2, Budget_Amount
- 🆕 **Intelligent text parsing**: Extracts amounts even from unstructured text
- 🆕 **Multi-format support**: PDF tables, Excel sheets, image OCR
- 🆕 **Smart amount detection**: Recognizes various currency formats ($, €, £, ₹, etc.)
- 🆕 **Highest budget highlighting**: Automatically identifies highest budget entries
- 🆕 **Batch processing**: Process all files in blob storage at once

---

## Supported File Types

- **PDF files**: Tables and text-based budget information
- **Excel files** (.xlsx, .xls): All sheets processed automatically  
- **Image files** (.jpg, .jpeg, .png, .tiff, .bmp): OCR-based extraction
- **Text formats**: Recognizes budget amounts in plain text

---

## Requirements

- Python 3.8+
- Azure Document Intelligence (Form Recognizer) resource
- Azure Blob Storage account
- Required Python packages (see requirements.txt)

Install dependencies:
```sh
pip install -r requirements.txt
```

---

## Setup

1. **Clone or download this repository.**

2. **Create a `.env` file in the project directory:**
   ```
   # Azure Document Intelligence Configuration
   DOC_INTELLIGENCE_ENDPOINT=https://your-document-intelligence-resource.cognitiveservices.azure.com/
   DOC_INTELLIGENCE_KEY=your-document-intelligence-key-here

   # Azure Blob Storage Configuration  
   AZURE_BLOB_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=your-storage-account;AccountKey=your-account-key;EndpointSuffix=core.windows.net
   AZURE_BLOB_CONTAINER=your-container-name
   ```

3. **Upload files to Azure Blob Storage:**
   - Upload your PDF, Excel, or image files to the specified blob container

4. **Run the app:**
   ```sh
   streamlit run main.py
   ```

---

## Usage

### Budget Extraction Mode

1. **Select "Budget Extraction" or "Both"** from the processing mode options
2. **Choose a file** from your blob storage or process all files
3. **View results** in standardized 3-column format:
   - **Label1**: Primary label/description
   - **Label2**: Secondary label/category  
   - **Budget_Amount**: Extracted monetary amount

### Key Features of Budget Extraction

- **Smart Currency Detection**: Recognizes $, €, £, ₹, USD, EUR, GBP, INR, etc.
- **Text Format Support**: Extracts amounts like "5,000", "$25,000", "Rs. 150,000"
- **Abbreviated Amounts**: Handles "25K", "1.5M", "2B" formats
- **Highest Budget Identification**: Automatically highlights the largest amount
- **Multi-source Processing**: Combines data from tables, text, and multiple files

### Example Output

| Label1 | Label2 | Budget_Amount |
|--------|--------|---------------|
| Marketing | Q4 Campaign | 25000.00 |
| Equipment | New Laptops | 150000.00 |
| Training | Employee Development | 8750.00 |

---

## New Files Added

- `budget_extractor.py`: Core budget extraction logic
- `test_budget_extractor.py`: Test suite for budget functionality
- `.env.template`: Template for environment variables
- Updated `main.py`: Integrated budget extraction UI
- Updated `requirements.txt`: Additional dependencies

---

## Usage

1. Open the app in your browser (Streamlit will show the URL).
2. Upload a PDF document.
3. Click **"Extract Tables from Documents"**.
4. View tables containing monetary values.
5. Download all tables as CSV or JSON.

---

## Notes

- The app uses the **prebuilt-layout** model for table extraction.
- Only tables containing monetary values (amount, price, total, etc.) are shown.
- You need an active Azure Document Intelligence resource.

---

## Troubleshooting

- **Missing endpoint or key:**  
  Make sure your `.env` file is present and contains the correct values.

- **No tables found:**  
  The PDF may not contain tables or monetary values recognizable by the model.

- **Azure errors:**  
  Check your Azure subscription, endpoint, and key.

---

## License

This project is for demonstration and educational purposes.

---

## Credits

- [Azure Document Intelligence (Form Recognizer)](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/)
-