# src/inference.py nâng cấp

import os
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from src.preprocess import full_preprocess, extract_lexicon_features, extract_stat_features
from src.config import DEFAULT_THRESHOLD_HATE, DEFAULT_THRESHOLD_OFF

class ViHSDModel:
    def __init__(self, model_path='outputs/results_training/vihsd_pipeline.pkl'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy model tại: {model_path}")
        
        self.pipeline = joblib.load(model_path)
        self.tfidf = self.pipeline['tfidf']
        self.count_vec = self.pipeline['count_vec']
        self.lda = self.pipeline['lda']
        self.selector = self.pipeline['selector']
        self.model = self.pipeline['best_model']
        self.label_map = self.pipeline['label_map']
        self.use_tokenizer = self.pipeline.get('use_tokenizer', True)

    def predict(self, text, t_hate=DEFAULT_THRESHOLD_HATE, t_off=DEFAULT_THRESHOLD_OFF):
        # 1. Tiền xử lý
        cleaned = full_preprocess(text, use_tokenizer=self.use_tokenizer)
        if not cleaned.strip(): cleaned = "empty"
            
        # 2. Trích xuất đặc trưng
        X_tf = self.tfidf.transform([cleaned])
        X_ld = self.lda.transform(self.count_vec.transform([cleaned]))
        X_lx = extract_lexicon_features([cleaned])
        
        temp_df = pd.DataFrame({'text': [text], 'text_clean': [cleaned]})
        X_st = extract_stat_features(temp_df)
        
        # 3. Ghép và chọn đặc trưng
        X_f = hstack([X_tf, csr_matrix(X_ld), csr_matrix(X_lx), csr_matrix(X_st)])
        X_s = self.selector.transform(X_f)
        
        # 4. Dự đoán với Threshold Tuning (Cốt lõi của sự cải thiện)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_s)[0] # [prob_clean, prob_off, prob_hate]
            
            # Áp dụng logic ưu tiên nhãn độc hại
            if probs[2] >= t_hate:
                pred_idx = 2
            elif probs[1] >= t_off:
                pred_idx = 1
            else:
                pred_idx = 0
            
            confidence = float(probs[pred_idx])
        else:
            # Dự phòng nếu model không có proba
            pred_idx = int(self.model.predict(X_s)[0])
            confidence = 1.0
            
        return {
            'label': self.label_map[pred_idx],
            'confidence': confidence,
            'cleaned_text': cleaned
        }