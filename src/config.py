# src/config.py
import os

# Nạp biến môi trường từ file .env (nếu có python-dotenv).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- CẤU HÌNH MODEL REMOTE (ViHateT5 host trên Kaggle qua ngrok) ---
# URL ngrok đổi mỗi phiên Kaggle -> luôn đọc từ .env, không hardcode.
REMOTE_API_URL = os.environ.get("REMOTE_API_URL", "").strip().rstrip("/")
# Tự thêm scheme nếu người dùng quên (vd dán "xxx.ngrok-free.dev").
if REMOTE_API_URL and not REMOTE_API_URL.startswith(("http://", "https://")):
    REMOTE_API_URL = "https://" + REMOTE_API_URL
REMOTE_API_TOKEN   = os.environ.get("REMOTE_API_TOKEN", "")
REMOTE_API_TIMEOUT = float(os.environ.get("REMOTE_API_TIMEOUT", "60"))

# --- Bổ sung vào cuối file config.py ---
# Ngưỡng tối ưu tìm được từ quá trình Tuning
DEFAULT_THRESHOLD_HATE = 0.35
DEFAULT_THRESHOLD_OFF  = 0.35
# CẤU HÌNH CHUNG 
SEED = 42
LABEL_NAMES = ['CLEAN', 'OFFENSIVE', 'HATE']
LABEL_MAP = {0: 'CLEAN', 1: 'OFFENSIVE', 2: 'HATE'}

# Cờ báo hiệu có sử dụng thư viện tách từ tiếng Việt hay không (Đồng bộ với lúc train)
USE_VN_TOKENIZER = True

#TỪ ĐIỂN CHUẨN HÓA (SUPER DICT)
TEEN_CODE_DICT = {
    # PROFANITY
    "dm": "dit me", "dmm": "dit me", "dmmm": "dit me", "dmn": "dit me", "dmnn": "dit me",
    "dcm": "dit con me", "dkm": "dit me", "duma": "dit me",
    "cl": "lon", "clm": "cai lon me", "clmm": "cai lon me",
    "vl": "vai lon", "vcl": "vai ca lon", "vkl": "vai lon", "vlon": "vai lon",
    "cc": "con cac", "cmm": "con me may", "ccm": "con cac me",
    "lol": "lon", "loz": "lon", "l": "lon", "lz": "lon",
    "dit": "dit", "lon": "lon", "cac": "cac",
    "tml": "to me lon",
    
    # EXTREME SLANG
    "vlll": "vai lon", "vllll": "vai lon", "vcll": "vai ca lon",
    "dmmmm": "dit me", "cllll": "lon", "ccccc": "con cac",
    
    # KHÔNG
    "ko": "khong", "k": "khong", "kh": "khong",
    "hok": "khong", "hông": "khong", "khum": "khong", "hem": "khong",
    "k0": "khong", "kg": "khong", "hong": "khong",
    
    # NGỮ PHÁP / NỐI TỪ
    "dc": "duoc", "đc": "duoc", "dk": "duoc", "đk": "duoc",
    "vs": "voi", "v": "voi",
    "r": "roi", "roi": "roi", "roii": "roi", "ròi": "roi", "rùi": "roi",
    "nma": "nhung ma", "nmaaa": "nhung ma", "nhma": "nhung ma",
    "cx": "cung", "cug": "cung", "cũng": "cung",
    "ms": "moi",
    
    # ĐẠI TỪ
    "t": "toi", "mk": "minh", "mik": "minh", "m": "may",
    "b": "ban", "ae": "anh em", "mn": "moi nguoi", "tụi": "chung toi",
    
    # TIẾNG ANH LÓNG
    "wtf": "cai quai", "omg": "oi troi",
    "lolol": "lon", "lmao": "cuoi chet", "bruh": "that a",
    
    # TÍCH CỰC & KHÁC
    "ok": "tot", "oke": "on", "okie": "on", "okela": "on", "oce": "on",
    "good": "tot", "nice": "tot",
    "bt": "binh thuong", "nx": "nua", "ns": "noi",
    "tks": "cam on", "thx": "cam on", "pls": "lam on", "plz": "lam on"
}

#STOPWORDS AN TOÀN 
VN_STOPWORDS = {
    "va", "la", "cua", "co", "trong", "da", "duoc", "cho", "voi",
    "cac", "mot", "nhung", "nay", "do", "nguoi", "ve", "hay", "thi",
    "tu", "nhu", "con", "khi", "vi", "cung", "ma", "den", "lai",
    "ra", "di", "len", "xuong", "vao", "qua", "tren", "duoi",
    "nen", "theo", "tai", "the", "cai",
    "a", "oi", "u", "uh", "uhm", "um", "ah",
    "nhe", "nhi", "thoi", "ha", "he", "ne",
    "nao", "gi", "ai", "sao", "day", "kia",
    "bi", "boi", "do", "de", "lam", "noi", "biet",
    "that", "thuc", "su", "vay", "thay",
    "ahh", "uhh", "hmm", "huhu", "haha", "hehe", "kkk", "kk", "hi", "hihi",
    "coi", "xem"
}

# Tập hợp các từ độc hại dùng cho Lexicon Features (Lúc dự đoán sẽ cần cái này)
TOXIC_WORDS = set([
    'chet', 'giet', 'ngu', 'dan', 'vo hoc', 'suc vat', 'thoi', 'hen', 'khon', 'ban', 'rac',
    'may', 'tao', 'chung may', 'bon', 'lu', 'do', 'thang', 'con',
    'ghet', 'kinh', 'cam', 'thu', 'tom', 'xau xa',
] + list(TEEN_CODE_DICT.keys()))