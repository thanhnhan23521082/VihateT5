# src/preprocess.py

import re
import unicodedata
import pandas as pd
import numpy as np
from underthesea import word_tokenize as vn_tokenize

# Import cấu hình từ file config
from src.config import TEEN_CODE_DICT, VN_STOPWORDS, TOXIC_WORDS

def normalize_unicode(text):
    text = unicodedata.normalize('NFC', str(text))
    return text.lower().strip()

def remove_noise(text):
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    emoji_pattern = re.compile(
        '[\U00010000-\U0010ffff\U0001F300-\U0001F9FF\u2600-\u26FF\u2700-\u27BF]+',
        flags=re.UNICODE)
    return emoji_pattern.sub(' ', text)

def clean_special_chars(text):
    text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def normalize_repeat(text):
    return re.sub(r'(.)\1{2,}', r'\1', text)

def normalize_text_regex(text):
    text = re.sub(r'\bd+m+\b', 'dm', text)
    text = re.sub(r'\bv+l+\b', 'vl', text)
    text = re.sub(r'\bc+c+\b', 'cc', text)
    text = re.sub(r'\bd+k+m+\b', 'dkm', text)
    text = re.sub(r'\bv+c+l+\b', 'vcl', text)
    text = re.sub(r'\bc+l+m+\b', 'clm', text)
    return text

def expand_teen_code(text):
    words = text.split()
    return ' '.join([TEEN_CODE_DICT.get(w, w) for w in words])

def vn_tokenize_text(text):
    try:
        return vn_tokenize(text, format='text')
    except Exception:
        return text

def remove_stopwords(text):
    words = text.split()
    return ' '.join([w for w in words if w not in VN_STOPWORDS and len(w) > 1])

def full_preprocess(text, use_tokenizer=True):
    text = normalize_unicode(text)
    text = remove_noise(text)
    text = clean_special_chars(text)
    text = normalize_repeat(text)
    text = normalize_text_regex(text)
    text = expand_teen_code(text)
    if use_tokenizer:
        text = vn_tokenize_text(text)
    text = remove_stopwords(text)
    return text.strip()


def extract_lexicon_features(texts):
    feats = []
    for text in texts:
        words = str(text).lower().split()
        toxic_cnt = sum(1 for w in words if w in TOXIC_WORDS)
        total_words = max(len(words), 1)
        feats.append([toxic_cnt, toxic_cnt / total_words, sum(1 for w in words if len(w) > 8)])
    return np.array(feats, dtype=np.float32)

def extract_stat_features(df):
    feats = pd.DataFrame()
    text       = df['text'].astype(str)
    text_clean = df['text_clean'].fillna('').astype(str)
    
    feats['char_len']       = text.apply(len)
    feats['word_count']     = text_clean.apply(lambda x: len(x.split()))
    feats['avg_word_len']   = text_clean.apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0)
    feats['exclaim_count']  = text.str.count(r'!')
    feats['question_count'] = text.str.count(r'\?')
    feats['caps_ratio']     = text.apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
    feats['digit_ratio']    = text.apply(lambda x: sum(1 for c in x if c.isdigit()) / max(len(x), 1))
    feats['has_url']        = text.str.contains(r'https?://', regex=True).astype(int)
    feats['has_mention']    = text.str.contains(r'@\w+', regex=True).astype(int)
    feats['repeat_char']    = text.str.contains(r'(.)\1{2,}', regex=True).astype(int)
    
    return feats.values.astype(np.float32)