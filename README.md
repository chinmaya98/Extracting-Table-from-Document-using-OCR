# PDF Table Extractor with Azure Document Intelligence

This Streamlit app allows you to **upload a PDF document**, extract tables using Azure Document Intelligence OCR (prebuilt-layout model), and download the results as CSV or JSON.  
It automatically filters tables containing monetary values (amounts, prices, totals, etc.).

---

## Features

- Upload PDF files locally (no Azure Blob Storage required)
- Extract tables using Azure Document Intelligence (Form Recognizer)
- Filter tables containing money/amounts
- View extracted tables in the browser
- Download all tables as CSV or JSON

---

## Requirements

- Python 3.8+
- Azure Document Intelligence (Form Recognizer) resource (endpoint & key)
- The following Python packages:
  - `streamlit`
  - `pandas`
  - `python-dotenv`
  - `azure-ai-documentintelligence`

Install dependencies:
```sh
pip install streamlit pandas python-dotenv azure-ai-documentintelligence
```

---

## Setup

1. **Clone or download this repository.**

2. **Create a `.env` file in the project directory:**
   ```
   DOC_INTELLIGENCE_ENDPOINT=https://<your-resource-name>.cognitiveservices.azure.com/
   DOC_INTELLIGENCE_KEY=<your-document-intelligence-key>
   ```
   Replace `<your-resource-name>` and `<your-document-intelligence-key>` with your actual Azure resource values.

3. **Run the app:**
   ```sh
   streamlit run main.py
   ```

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

