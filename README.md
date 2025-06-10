# German Insurance Document Extractor

A Streamlit application that uses Google's Gemini AI model to extract structured information from German insurance documents.

## Features

- PDF document processing
- AI-powered information extraction
- Structured JSON output
- User-friendly interface

## Requirements

- Python 3.8+
- Streamlit
- Google Generative AI
- PyMuPDF (fitz)
- Pillow
- pandas
- python-dotenv

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd insurance-document-extractor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Google AI API key:
```
GENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run main.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Upload a German insurance document (PDF format)

4. Click "Extract Document Fields" to process the document

## Output

The application provides two types of output:
1. Extracted Fields: Raw JSON containing the extracted information
2. Structured JSON Output: Formatted JSON with additional metadata

## License

MIT License 