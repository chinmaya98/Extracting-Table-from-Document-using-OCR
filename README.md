# Trinity Online - AI Enhanced Budget Extractor

Trinity Online is an intelligent budget table extraction system that uses **Azure Document Intelligence** and **GPT 4.1** to automatically process documents and extract structured budget data. The application features a specialized architecture with dedicated processors for different file types.

**🤖 NEW: Specialized Architecture** - Refactored into dedicated processors for optimal performance and maintainability.

---

## 🏗️ Architecture Overview

Trinity Online uses a **specialized processor architecture** to handle different file types optimally:

```
📁 Input Files
├── PDF/Images → table_extractor_pdf_image.py → Document Intelligence OCR → AI Processing
├── Excel Files → table_extractor_excel.py → Direct Processing → AI Processing
└── All Files → ai_budget_extractor.py → Structured 3-Column Output
```

### 🔄 Data Flow

```
1. File Upload (Azure Blob Storage)
   ↓
2. File Type Detection
   ↓
3A. PDF/Image Path:                    3B. Excel Path:
    - Document Intelligence OCR            - Direct sheet reading
    - Table structure extraction           - Individual sheet processing
    - Original header preservation         - Original header preservation
   ↓                                      ↓
4. AI Budget Extractor (GPT 4.1)
   - Budget table identification
   - Smart formatting and analysis
   - Bold headers and totals
   ↓
5. Export Options (CSV, Excel, JSON)
```

---

## � Core Components

### 1. **table_extractor_pdf_image.py**
**Purpose**: Specialized processor for PDF and image files
- 🔍 Uses Azure Document Intelligence Layout Model for OCR
- 📄 Extracts tables from PDF pages and image files
- 🏷️ Preserves original headers from source documents
- 💰 Filters budget-related tables automatically
- 📊 Passes structured data to AI Budget Extractor

**Supported Formats**: `.pdf`, `.jpg`, `.jpeg`, `.png`, `.tiff`, `.bmp`

### 2. **table_extractor_excel.py**
**Purpose**: Specialized processor for Excel files
- � Direct Excel file processing (no Document Intelligence needed)
- 📋 Processes each sheet individually in the workbook
- 🏷️ Preserves original Excel column headers
- 💰 Filters budget-relevant sheets automatically
- 🔄 Supports both `.xlsx` and `.xls` formats

**Key Features**:
- Individual sheet iteration and analysis
- Multi-engine support (openpyxl, xlrd)
- Original header preservation per sheet
- Sheet metadata tracking

### 3. **ai_budget_extractor.py**
**Purpose**: AI-enhanced budget processing engine
- 🤖 GPT 4.1 powered intelligent analysis
- 🏷️ Dynamic header extraction and bold formatting
- 💡 Context-aware budget table identification
- 📊 Standardized 3-column output generation
- 🧠 AI Budget Summary generation for non-budget documents
- 🔄 Fallback processing when AI is unavailable

### 4. **ui_app.py**
**Purpose**: Trinity Online user interface
- 🖥️ Clean, professional Streamlit interface
- 📁 Multi-sheet display with individual tables
- 💾 Export options: CSV, Excel, JSON
- 📊 Real-time processing progress
- 🎯 Source information and sheet tracking

---

## Features

### 🤖 AI-Enhanced Processing
- **GPT 4.1 Integration**: Azure OpenAI for intelligent budget analysis
- **Original Headers**: Preserves exact headers from source documents
- **Smart Formatting**: Automatic bold headers and totals
- **Context Analysis**: Uses document context for better understanding

### 🧠 AI Budget Summary (NEW)
- **Always-On Analysis**: Every document gets an AI-powered budget summary using GPT 4.1
- **Budget-Focused**: Specifically analyzes content for financial, monetary, and budget information
- **Intelligent Insights**: Provides 2-3 line summaries highlighting budget relevance
- **No More Errors**: Replaces generic error messages with meaningful budget analysis
- **Prominent Display**: Summary appears first, before any table processing

### 📊 Multi-Sheet Excel Support
- **Individual Processing**: Each Excel sheet processed separately
- **Original Headers**: Preserves Excel column names exactly
- **Sheet Identification**: Clear tracking of source sheet names
- **Flexible Export**: Download individual sheets or combined files

### 💾 Comprehensive Export Options
- **CSV**: Individual sheets and combined files
- **Excel**: Bold headers with individual sheet tabs
- **JSON**: Structured data for API integration
- **Original Headers**: All exports preserve source document headers

---

## Supported File Types

| Format | Processor | Processing Method | Headers Source |
|--------|-----------|------------------|----------------|
| **PDF** | PDF/Image Extractor | Document Intelligence OCR | Original document tables |
| **Images** | PDF/Image Extractor | Document Intelligence OCR | Original image tables |
| **Excel (.xlsx/.xls)** | Excel Extractor | Direct sheet reading | Excel column headers |

---

## Requirements

- Python 3.8+
- Azure Document Intelligence resource (for PDF/Images)
- Azure Blob Storage account
- Azure OpenAI GPT 4.1 deployment
- Required Python packages (see requirements.txt)

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Setup

1. **Clone or download this repository.**

2. **Create a `.env` file in the project directory:**
   ```bash
   # Azure Document Intelligence Configuration
   DOC_INTELLIGENCE_ENDPOINT=https://your-document-intelligence-resource.cognitiveservices.azure.com/
   DOC_INTELLIGENCE_KEY=your-document-intelligence-key-here

   # Azure Blob Storage Configuration  
   AZURE_BLOB_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=your-storage-account;AccountKey=your-account-key;EndpointSuffix=core.windows.net
   AZURE_BLOB_CONTAINER=your-container-name

   # Azure OpenAI GPT 4.1 Configuration
   AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/
   AZURE_OPENAI_API_KEY=your-openai-key-here
   AZURE_OPENAI_API_VERSION=2025-01-01-preview
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1
   ```

3. **Upload files to Azure Blob Storage:**
   - Upload your PDF, Excel, or image files to the specified blob container

4. **Run Trinity Online:**
   ```bash
   python -m streamlit run main.py
   ```

   or

   ```bash
   streamlit run main.py
   ```

5. **Access the application:**
   - Open your browser to `http://localhost:8501`
   - The Trinity Online interface will be available

---

## Usage

### Trinity Online Interface

1. **Select a file** from your Azure Blob Storage using the dropdown
2. **Click "Process Selected File"** to start the Trinity workflow
3. **Monitor progress** through the 5-step process:
   - File download from blob storage
   - Document Intelligence processing
   - AI budget analysis with GPT 4.1
   - Data formatting and cleanup
   - Final result preparation
4. **View results** in the standardized table format
5. **Download** your data as CSV or Excel file

### Output Format

The AI system produces clean, standardized budget data:

```
Label                          Description                      Amount
Salary            Annual salary for Q4 2024                 45000.0
Marketing         Digital advertising for Q4 2024          15000.0
Travel            Business trips during Q4 2024             8750.0
Benefits          Health insurance for staff                12000.0
Equipment         Laptops and monitors for office use       8500.0
Office Supplies   Stationery and materials for Q4 2024      2500.0
**TOTAL**         Total budget allocation for Q4 2024      91750.0
```

### Key Features
- **Smart Currency Detection**: Recognizes various currency formats
- **Intelligent Totaling**: Automatically calculates and highlights totals
- **Context Integration**: Uses document text for better understanding
- **Professional Formatting**: Bold headers and total rows

---

## Troubleshooting

### Common Issues

**Application failed to start:**
- Check that all Azure services are properly configured in `.env`
- Ensure Azure Document Intelligence and OpenAI endpoints are correct
- Verify API keys are valid and not expired

**No files found in blob storage:**
- Confirm blob storage connection string is correct
- Check that files are uploaded to the specified container
- Verify container name matches the configuration

**AI processing errors:**
- Ensure GPT 4.1 deployment is active and accessible
- Check Azure OpenAI API key and endpoint
- Verify API version is supported (2025-01-01-preview)

**Document processing failures:**
- Confirm Document Intelligence service is running
- Check file format is supported (PDF, Excel, Images)
- Verify endpoint and key are correctly configured

### Testing Commands

**Test imports:**
```bash
python -c "import ui_app; print('Trinity UI App imported successfully!')"
```

**Run Trinity Online:**
```bash
python -m streamlit run main.py
```

---

## Technical Notes

- **AI Processing**: Works on structured Document Intelligence data, not raw files
- **Fallback Support**: Automatic fallback to standard extraction if AI fails  
- **Multi-format Support**: PDF, Excel (.xlsx, .xls), Images (.jpg, .png, .tiff, .bmp)
- **Professional Output**: Clean 3-column format with bold totals and headers
- **Real-time Processing**: Live progress indicators during workflow execution

---

## License

This project is for demonstration and educational purposes.

---

## Credits

- [Azure Document Intelligence](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/)
- [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Streamlit](https://streamlit.io/) - Web application framework