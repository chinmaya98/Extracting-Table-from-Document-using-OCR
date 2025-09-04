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

## Prerequisites

- Python 3.11+
- Docker (for container deployment)
- Azure account with access to:
  - Azure Container Registry (ACR)
  - Azure App Service (Web App for Containers)
  - Azure Document Intelligence resource
  - Azure Blob Storage

---

## Local Development

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd Trinity_online
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Create a `.env` file in the project root with your Azure credentials:
     ```
     DOC_INTELLIGENCE_ENDPOINT=your-endpoint
     DOC_INTELLIGENCE_KEY=your-key
     AZURE_BLOB_CONNECTION_STRING=your-connection-string
     AZURE_BLOB_CONTAINER=your-container-name
     ```

4. **Run the app:**
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
