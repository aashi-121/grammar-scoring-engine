# 🎙️ Grammar Scoring Engine from Voice Samples

## 📌 Project Overview
- This project implements an end-to-end Grammar Scoring Engine that evaluates spoken English responses by analyzing audio input, converting it into text, extracting linguistic and grammatical features, and training a machine learning model to predict grammar quality scores.

- The solution demonstrates the complete pipeline from raw voice data → transcription → NLP-based feature extraction → ML model training, designed to be scalable and interpretable.

## 🧠 Problem Statement
Given short voice recordings of spoken English, the goal is to automatically assess the grammar quality of the speech and assign a numerical score.

### Key challenges addressed:

- Handling raw audio data
- Accurate speech-to-text conversion
- Extracting meaningful grammar-related features
- Building a regression-based scoring model

## ⚙️ Tech Stack
- Python 3.11
- OpenAI Whisper – Speech-to-text transcription
- Librosa – Audio loading & preprocessing
- spaCy – NLP & grammatical feature extraction
- scikit-learn – Model training & evaluation
- pandas / numpy – Data handling
- Git & GitHub – Version control

## 🏗️ System Architecture
Audio (.wav)
   ↓
Speech-to-Text (Whisper)
   ↓
Text Cleaning & Normalization
   ↓
Grammar Feature Extraction (NLP)
   ↓
Feature Vector
   ↓
ML Regression Model
   ↓
Grammar Score


## 🔍 Features Extracted
For each transcribed response, the following features are computed:

- Number of words
- Average word length
- Number of sentences
- Number of verbs (POS-based)

These features were chosen for:

- Interpretability
- Low computational overhead
- Direct relevance to grammatical structure

## 🤖 Model Details
Model Used: Linear Regression

### Reason:

- Simple and interpretable baseline
- Suitable for structured numerical features
- Evaluation Metric: Mean Absolute Error (MAE)

The model is trained on extracted features from training audio samples and validated on a held-out split.


## 📂 Project Structure
```
grammar-scoring-engine/
│
├── main.py              # Complete pipeline: audio → score
├── requirements.txt     # Project dependencies
├── .gitignore           # Prevents dataset & credentials upload
├── README.md            # Project documentation
│
├── dataset/             # (Ignored) Audio & CSV files
│   ├── audios/
│   └── csvs/
│
└── submission.csv       # (Ignored) Generated predictions
```

## 🚀 How to Run
- 1️⃣ Install dependencies
```
pip install -r requirements.txt
```
- 2️⃣ Run the pipeline
```
python main.py
```
### The script will:
- Load audio
- Transcribe speech
- Extract grammar features
- Train a model

## Output evaluation metrics

### 🧪 Key Learnings
- Practical handling of real-world audio data
- Trade-offs between accuracy and computation in speech models
- Importance of clean feature engineering in ML pipelines
- Managing large datasets safely using .gitignore


## 🔐 Data & Ethics
- Dataset used only for assessment purposes
- No raw audio or labeled data uploaded to GitHub
- All sensitive files are excluded via .gitignore



## 📈 Future Improvements
- Use transformer-based grammar evaluation models
- Add pronunciation and fluency features
- Try ensemble or tree-based regressors
- Deploy as a REST API for real-time scoring



## 👤 Author
Aashi Sharma
Computer Science Engineering Student













