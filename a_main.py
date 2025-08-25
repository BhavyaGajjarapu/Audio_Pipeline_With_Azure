import os, json, re, tempfile
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import whisper
import torch
import librosa
import numpy as np
from pydub import AudioSegment
from panns_inference import AudioTagging
from sentence_transformers import SentenceTransformer, util
import spacy
from transformers import pipeline
from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline
from datetime import datetime, timezone
import string

HF_TOKEN = os.getenv("HF_TOKEN")

# === Azure Storage ===
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
load_dotenv()
AZURE_CONNECT_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "audiofiles")
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECT_STR)

# === FFMPEG Setup ===
AudioSegment.converter = "ffmpeg"  # Assuming ffmpeg in PATH

# === Models ===
model_embedder = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")
whisper_model = whisper.load_model("small.en")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

# === Diarization ===
try:
	diarization_pipeline = Pipeline.from_pretrained(
		"pyannote/speaker-diarization", use_auth_token=HF_TOKEN
	)
except Exception as e:
	diarization_pipeline = None
	print(f"[WARN] Diarization model not loaded: {e}")

def run_diarization(path):
	"""Runs speaker diarization and returns speaker segments."""
	if not diarization_pipeline:
		return "Diarization model not available"
	diarization = diarization_pipeline(path)
	segments = []
	for turn, _, speaker in diarization.itertracks(yield_label=True):
		segments.append({
			"speaker": speaker,
			"start": round(turn.start, 2),
			"end": round(turn.end, 2)
		})
	return segments

# === Summarization ===
try:
	summarizer = hf_pipeline("summarization", model="facebook/bart-large-cnn")
except Exception as e:
	summarizer = None
	print(f"[WARN] Summarizer model not loaded: {e}")

def generate_summary(text):
	if not summarizer:
		return "Summarizer model not available"
	try:
		text_len = len(text.split())
		# Use adaptive lengths but with minimum limits
		max_len = max(20, min(60, int(text_len * 0.4)))
		min_len = max(10, int(max_len * 0.5))
		# If transcript is very short, set max_len to a small fixed number (e.g. 20)
		if text_len < 20:
			max_len = 20
			min_len = 10
		return summarizer(
			text,
			max_length=max_len,
			min_length=min_len,
			do_sample=False
		)[0]["summary_text"]
	except Exception as e:
		return f"Summary failed: {e}"

# === Simple Action Item Extraction ===
def extract_action_items(text):
	"""Very basic action item extraction."""
	lines = text.split(". ")
	action_items = [line for line in lines if any(word in line.lower() for word in ["need to", "should", "must", "let's", "action", "plan"])]
	return action_items if action_items else "No clear action items found."

# === Audio Tagging ===
def detect_audio_tags(path):
	import csv
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	tagger = AudioTagging(device=device)
	waveform, _ = librosa.load(path, sr=32000, mono=True)
	waveform = waveform[None, :]
	result = tagger.inference(waveform)
	output = result[0]
	# FIXED: Ensure output is a torch tensor before .detach()
	if isinstance(output, np.ndarray):
		probs = output  # already NumPy
	else:
		probs = output.detach().cpu().numpy()
	with open("C:/Users/Bhavya Gajjarapu/panns_data/class_labels_indices.csv") as f:
		reader = csv.DictReader(f)
		class_names = [r["display_name"] for r in reader]
	probs = probs[0] if probs.ndim > 1 else probs
	top_indices = np.argsort(probs)[::-1][:5]
	return [(class_names[i], float(probs[i])) for i in top_indices]

# === Transcription ===
def transcribe_with_timestamps(path):
	result = whisper_model.transcribe(path, language="en", fp16=False, word_timestamps=True)
	word_data = []
	for segment in result["segments"]:
		for word in segment.get("words", []):
			word_data.append({"word": word["word"], "start": word["start"], "end": word["end"]})
	return result["text"], word_data

# === NLP Functions ===
def emotion_recognition(path):
	tags = detect_audio_tags(path)
	emo = [t for t, _ in tags if any(e in t.lower() for e in ["happy", "sad", "angry", "fear"])]
	return emo[0] if emo else "neutral"

def analyze_text(text):
	doc = nlp(text)
	return [(ent.text, ent.label_) for ent in doc.ents]

def analyze_tones_local(text):
	try:
		emotion_preds = emotion_classifier(text[:512])[0]
		return sorted(emotion_preds, key=lambda x: x['score'], reverse=True)[:3]
	except Exception:
		return []

# === Azure File Download & Processing ===
def download_blob_to_tempfile(blob_name):
	blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=blob_name)
	download_stream = blob_client.download_blob()
	tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
	tmp_file.write(download_stream.readall())
	tmp_file.close()
	return tmp_file.name

from spellchecker import SpellChecker
spell = SpellChecker()

def autocorrect_query(query: str) -> str:
	"""Auto-correct words in the query using pyspellchecker."""
	words = query.split()
	corrected_words = []
	for word in words:
		# Skip numbers and very short words
		if word.isnumeric() or len(word) <= 2:
			corrected_words.append(word)
			continue
		corrected_word = spell.correction(word)
		corrected_words.append(corrected_word if corrected_word else word)
	corrected_query = " ".join(corrected_words)
	return corrected_query

# === Main Audio Processor ===
def process_audio_blob(blob_name):
	temp_path = download_blob_to_tempfile(blob_name)
	file_id = os.path.splitext(os.path.basename(temp_path))[0]
	audio = AudioSegment.from_file(temp_path).set_frame_rate(16000).set_channels(1)
	wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
	audio.export(wav_path, format="wav")
	tags = detect_audio_tags(wav_path)
	transcript, word_data = transcribe_with_timestamps(wav_path)
	emotion = emotion_recognition(wav_path)
	entities = analyze_text(transcript)
	tones = analyze_tones_local(transcript)
	diarization = run_diarization(wav_path)  # new
	summary = generate_summary(transcript)  # new
	action_items = extract_action_items(transcript)  # new
	return {
		"file_name": blob_name,
		"tags": [t for t, _ in tags],
		"tag_probs": {t: p for t, p in tags},
		"transcript": transcript,
		"emotion": emotion,
		"entities": entities,
		"tones": tones,
		"transcript_words": word_data,
		"diarization": diarization,
		"summary": summary,
		"action_items": action_items,
		"upload_time": datetime.now(timezone.utc).isoformat()
	}

# === Semantic Search ===
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

def semantic_search_audio_catalog(query, catalog, threshold=0.4):
	import re
	import string
	query_emb = model_embedder.encode(query, convert_to_tensor=True)
	results = []
	for item in catalog:
		combined_text = " ".join(item.get("tags", [])) + " " + item.get("transcript", "")
		combined_emb = model_embedder.encode(combined_text, convert_to_tensor=True)
		similarity = util.cos_sim(query_emb, combined_emb).item()
		if similarity > threshold:
			matched_timestamp = None
			if "transcript_words" in item:
				# Extract lowercased, non-stopword query words
				query_words = set(
					word for word in re.findall(r'\b\w+\b', query.lower())
					if word not in STOPWORDS
				)
				for word_data in item["transcript_words"]:
					word_clean = word_data["word"].lower().strip(string.punctuation + " ")
					if word_clean in query_words:
						matched_timestamp = word_data.get("start")
						break
			item = dict(item)
			item["match_score"] = similarity
			item["matched"] = query
			item["timestamp"] = round(matched_timestamp, 2) if matched_timestamp is not None else None
			results.append(item)
	results.sort(key=lambda x: x["match_score"], reverse=True)
	return results