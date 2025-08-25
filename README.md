Audio Pipeline with Azure

This project is a Flask-based web application for managing, processing, and searching audio files using Azure Blob Storage, NLP models, and speech processing pipelines.

🚀 Features

Audio Uploads

Upload audio files directly via the web interface

Files stored securely in Azure Blob Storage

Prevents duplicate uploads and logs all upload activities

Audio Processing Pipeline

Automatic speech-to-text transcription (Whisper)

Speaker diarization (Pyannote)

Emotion recognition (Audio tagging + NLP)

Entity extraction (SpaCy)

Tone analysis (Sentiment classifier)

Summarization (BART summarizer)

Action item extraction

Tags and metadata extraction for advanced search

Search

Semantic search across transcripts and tags using Sentence-Transformers

Autocorrect for queries using pyspellchecker

Search highlights exact timestamps in audio where matches occur

Results show streamable SAS URLs generated securely from Azure

Logging

Search queries logged in Audio_search_logging

Upload activities logged in Audio_upload_logging

All results/errors stored with timestamps for auditing

🛠️ Tech Stack

Backend: Flask (Python)

Database: SQL Server / Azure SQL (via pyodbc)

Storage: Azure Blob Storage

NLP Models:

Whisper (transcription)

Pyannote (speaker diarization)

Sentence-Transformers (all-MiniLM-L6-v2)

Hugging Face summarizer (facebook/bart-large-cnn)

Emotion classifier (j-hartmann/emotion-english-distilroberta-base)

SpaCy (en_core_web_sm)

Other Tools:

Librosa, Pydub, Torch, NumPy

PANNS Inference (Audio tagging)

Pyspellchecker

⚙️ Setup Instructions
1️⃣ Clone the Repository
git clone git@github.com:BhavyaGajjarapu/Audio_Pipeline_With_Azure.git
cd Audio_Pipeline_With_Azure

2️⃣ Create Virtual Environment & Install Dependencies
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

pip install -r requirements.txt

3️⃣ Configure Environment Variables

Create a .env file in the project root with:

# Azure
AZURE_ACCOUNT_NAME=your_account
AZURE_ACCOUNT_KEY=your_key
AZURE_CONTAINER_NAME=audiofiles
AZURE_STORAGE_CONNECTION_STRING=your_connection_string

# Database
DB_DRIVER={ODBC Driver 18 for SQL Server}
DB_HOST=your_sql_host
DB_PORT=1433
DB_NAME=your_db
DB_USER=your_user
DB_PASSWORD=your_password

# Hugging Face
HF_TOKEN=your_hf_token


⚠️ Do NOT commit your .env file (contains secrets). Add .env to .gitignore.

4️⃣ Run the Flask App
python a_app.py


Then visit: 👉 http://127.0.0.1:5000

📊 Database Tables

audio_metadata
Stores metadata for processed audio (transcript, summary, tags, entities, tones, etc.)

Audio_search_logging
Logs all user search queries with results/status.

Audio_upload_logging
Logs uploads with status (Success/Failure).

📂 Project Structure
Audio_Pipeline_With_Azure/
│── a_app.py          # Flask app, routes, DB, Azure logic
│── a_main.py         # Core audio processing & NLP pipeline
│── templates/
│   └── a_index.html  # Frontend template
│── static/
│   └── style.css     # Basic styling
│── .env              # (not committed) secrets
│── requirements.txt  # Dependencies
│── README.md         # Documentation

✨ Author: Bhavya Gajjarapu
🔗 GitHub: BhavyaGajjarapu