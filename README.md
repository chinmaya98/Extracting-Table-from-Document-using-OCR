# Trinity Online Table Extraction Application

This application extracts tables from PDF, image, Excel, and CSV files, performs budget analysis, and provides a user-friendly Streamlit web interface. It integrates with Azure Document Intelligence and Azure Blob Storage.

---

## Features

- Extract tables from PDF, images, Excel (.xls, .xlsx), and CSV files
- Budget data identification and extraction
- Azure Document Intelligence integration for document analysis
- Azure Blob Storage integration for file management
- Streamlit web UI for easy upload, viewing, and download of extracted tables
- Download tables as JSON

---


## System Requirements

### Software Requirements
- Python 3.8 or higher
- Internet connection (required for Azure services)
- Modern web browser (Chrome, Edge, Firefox, etc.) for accessing the Streamlit interface

### Azure Services Required
- **Azure Document Intelligence**: Used for OCR and document analysis
- **Azure Blob Storage**: Used for file storage and retrieval

---

## Installation & Setup

### 1. Environment Setup

Install required Python packages:
```sh
pip install -r requirements.txt
```

### 2. Dependencies Installation

The following packages will be installed:

- `streamlit>=1.28.0`          # Web interface framework
- `pandas>=1.5.0`              # Data manipulation
- `python-dotenv>=0.19.0`      # Environment variable management
- `azure-ai-documentintelligence>=1.0.0`  # Azure OCR services
- `azure-storage-blob>=12.0.0` # Azure file storage
- `openai>=0.28.0`             # AI processing capabilities
- `Pillow>=9.0.0`              # Image processing
- `filetype>=1.0.0`            # File type detection
- `regex>=2022.0.0`            # Pattern matching
- `PyPDF2>=3.0.0`              # PDF processing

---

## Local Development

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd Trinity_online
   ```

2. **(Optional) Create and activate a virtual environment:**
   ```sh
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Create a `.env` file in the project root with your Azure credentials:
     ```
     DOC_INTELLIGENCE_ENDPOINT=your-endpoint
     DOC_INTELLIGENCE_KEY=your-key
     AZURE_BLOB_CONNECTION_STRING=your-connection-string
     AZURE_BLOB_CONTAINER=your-container-name
     ```

5. **Run the app:**
   ```sh
   streamlit run main.py
   ```

---

## Docker Deployment

1. **Build the Docker image:**
   ```sh
   docker build -t trinity-online:latest .
   ```

2. **Tag and push to Azure Container Registry:**
   ```sh
   docker tag trinity-online:latest <your-acr-name>.azurecr.io/trinity-online:latest
   az acr login --name <your-acr-name>
   docker push <your-acr-name>.azurecr.io/trinity-online:latest
   ```

3. **Deploy to Azure App Service (Web App for Containers):**
   - Set the container image to `<your-acr-name>.azurecr.io/trinity-online:latest`
   - Add environment variables in App Service Configuration

---

## Usage

- Open the app in your browser (local: `http://localhost:8501`, Azure: `https://<your-app-service-name>.azurewebsites.net`)
- Upload PDF, image, Excel, or CSV files
- View extracted tables in expandable sections
- Download tables as JSON

---

## Troubleshooting

- **Docker errors:** Ensure Docker Desktop is running and WSL2 is installed/enabled.
- **Azure authentication:** Make sure you have AcrPush role for ACR and Contributor for App Service.
- **File type errors:** Only supported file types are accepted (`.pdf`, `.xlsx`, `.xls`, `.csv`, `.jpg`, `.jpeg`, `.png`, `.tiff`).

---

## Contact

For support, contact your development team or Azure
