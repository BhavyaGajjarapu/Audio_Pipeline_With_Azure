# Audio Pipeline with Azure

This project is a **Flask-based web application** for managing, processing, and searching audio files using **Azure Blob Storage**, **NLP models**, and **speech processing pipelines**.

---

## Features

### Audio Uploads
- Upload audio files directly via the web interface  
- Files stored securely in Azure Blob Storage  
- Prevents duplicate uploads and logs all upload activities  

### Audio Processing Pipeline
- Automatic speech-to-text transcription (**Whisper**)  
- Speaker diarization (**Pyannote**)  
- Emotion recognition (Audio tagging + NLP)  
- Entity extraction (**SpaCy**)  
- Tone analysis (**Sentiment classifier**)  
- Summarization (**BART summarizer**)  
- Action item extraction  
- Tags and metadata extraction for advanced search  

### Search
- Semantic search across transcripts and tags using **Sentence-Transformers**  
- Autocorrect for queries using **pyspellchecker**  
- Search highlights exact timestamps in audio where matches occur  
- Results show streamable **SAS URLs** generated securely from Azure  

### Logging
- Search queries logged in **Audio_search_logging**  
- Upload activities logged in **Audio_upload_logging**  
- All results/errors stored with timestamps for auditing  

---

## Tech Stack

**Backend:** Flask (Python)  
**Database:** SQL Server / Azure SQL (via pyodbc)  
**Storage:** Azure Blob Storage  

**NLP Models:**  
- Whisper (transcription)  
- Pyannote (speaker diarization)  
- Sentence-Transformers (all-MiniLM-L6-v2)  
- Hugging Face summarizer (facebook/bart-large-cnn)  
- Emotion classifier (j-hartmann/emotion-english-distilroberta-base)  
- SpaCy (en_core_web_sm)  

**Other Tools:**  
- Librosa, Pydub, Torch, NumPy  
- PANNS Inference (Audio tagging)  
- Pyspellchecker  

---

## 📊 Database Tables

- audio_metadata → Stores metadata for processed audio (transcript, summary, tags, entities, tones, etc.)

- Audio_search_logging → Logs all user search queries with results/status

- Audio_upload_logging → Logs uploads with status (Success/Failure)


## Project Structure
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