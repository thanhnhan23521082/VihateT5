# ViHateFilter: Vietnamese Hate Speech Detector

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Machine_Learning-scikit--learn-orange" alt="ML">
</p>

**ViHateFilter** (or *ViHSD*) is an end-to-end Machine Learning pipeline and REST API designed to detect, classify, and filter offensive language and hate speech in Vietnamese text. It features a lightweight backend powered by FastAPI and a modern, responsive web interface.

##  Key Features
- **Robust NLP Pipeline**: Custom text preprocessing including Teencode expansion, punctuation removal, and word segmentation using `underthesea`.
- **Machine Learning Engine**: An optimized classification model extracting lexical, statistical, and TF-IDF features to achieve a **0.73+ Macro F1** score on the ViHSD dataset.
- **RESTful API**: Fast and asynchronous endpoints built with FastAPI.
- **Modern Web Interface**: A sleek, user-friendly UI for quick real-time text analysis and batch testing visualization.

##  Project Structure

```text
├── data/               # Datasets (raw and processed) - Excluded from Git
├── notebooks/          # Jupyter notebooks for EDA and model training
├── outputs/            # Serialized models and evaluation plots
│   └── results_training/
│       └── vihsd_pipeline.pkl  # Core trained model
├── src/                # Core ML and NLP source code
│   ├── config.py       # Thresholds, dictionaries, and global settings
│   ├── preprocess.py   # Text normalization, slang expansion, tokenization
│   └── inference.py    # Model loading and prediction logic
├── static/             # Frontend assets (HTML, CSS, JS)
│   └── index.html      # Main Web Interface
├── app.py              # FastAPI application server
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

##  Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ViHateFilter.git
   cd ViHateFilter
   ```

2. **Set up a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure the model is ready:**
   Make sure the pre-trained model file `vihsd_pipeline.pkl` is located inside the `outputs/results_training/` directory. If not, run the training notebooks in `notebooks/` to generate it.

##  Running the Application

Start the FastAPI server equipped with the UI frontend:
```bash
uvicorn app:app --host 127.0.0.1 --port 8501 --reload
```
Once the server is running, open your web browser and navigate to:  
**[http://127.0.0.1:8501](http://127.0.0.1:8501)**

##  API Reference

### `GET /`
Serves the main web interface.

### `POST /api/predict`
Predicts the toxicity label of a given Vietnamese text.
- **Request Body (JSON):**
  ```json
  {
    "text": "Nội dung cần kiểm tra",
    "threshold": 0.5
  }
  ```
- **Response (JSON):**
  ```json
  {
    "label": "CLEAN | OFFENSIVE | HATE",
    "confidence": 0.98,
    "cleaned_text": "nội_dung cần kiểm_tra"
  }
  ```

##  License
This project is open-source and available under the [MIT License](LICENSE).
