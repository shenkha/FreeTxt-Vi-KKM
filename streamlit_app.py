import streamlit as st
import os # For path joining

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import nltk
from summa.summarizer import summarize as summa_summarizer # Added for English summarization
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import re
import pandas as pd
import string
from langdetect import detect, LangDetectException # Added for language detection

import argparse
# import os # Already imported above
import json
# import re # Already imported above
import unicodedata
from datetime import datetime

# import torch # Already imported above
# import nltk # Already imported above
import numpy as np
# import pandas as pd # Already imported above
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSeq2SeqLM,
    # AutoTokenizer, # Already imported above
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
import evaluate 

# --- Determine the absolute path to the script's directory ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STOPWORDS_PATH = os.path.join(SCRIPT_DIR, "vietnamese-stopwords.txt")

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

# Load Vietnamese stopwords
try:
    with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
        VIETNAMESE_STOPWORDS = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    st.error(f"vietnamese-stopwords.txt not found at {STOPWORDS_PATH}. Please ensure the file exists.")
    VIETNAMESE_STOPWORDS = []

# --- Summarization Model Loading (Cached) ---
@st.cache_resource
def load_summarization_model_vi(): # Renamed for clarity
    tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base-vietnews-summarization")
    model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base-vietnews-summarization")
    model.to(device)
    return tokenizer, model

# --- Sentiment Analysis Model Loading (Cached) ---
@st.cache_resource
def load_sa_model_vi(): # Renamed for clarity
    tokenizer = AutoTokenizer.from_pretrained("shenkha/FreeTxT-VisoBERT")
    model = AutoModelForSequenceClassification.from_pretrained("shenkha/FreeTxT-VisoBERT")
    model.to(device)
    return tokenizer, model

@st.cache_resource
def load_sa_model_en(): # For English SA
    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model.to(device)
    return tokenizer, model

# Load models 
SA_TOKENIZER_VI, SA_MODEL_VI = load_sa_model_vi()
SA_TOKENIZER_EN, SA_MODEL_EN = load_sa_model_en()
SUMM_TOKENIZER_VI, SUMM_MODEL_VI = load_summarization_model_vi()

SA_LABELS_VI = {
    0: "Tiêu cực", # Adjusted to match user's typical expectation for Vietnamese
    1: "Trung tính",
    2: "Tích cực"
}
SA_LABELS_EN = {
    0: "Very negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Very positive"
}

emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # Emoticons
    u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # Transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # Flags
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    u"\U0001f926-\U0001f937"
    u"\U00010000-\U0010ffff"
    u"\u200d"
    u"\u2640-\u2642"
    u"\u2600-\u2B55"
    u"\u23cf"
    u"\u23e9"
    u"\u231a"
    u"\u3030"
    u"\ufe0f"
    "]+", flags=re.UNICODE)

bang_nguyen_am = [['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
                  ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
                  ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
                  ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
                  ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
                  ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
                  ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
                  ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
                  ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
                  ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
                  ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
                  ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']]
bang_ky_tu_dau = ['', 'f', 's', 'r', 'x', 'j']

def chuan_hoa_dau_cau_tieng_viet(sentence):
    words = sentence.split()
    for index, word in enumerate(words):
        # Preserve surrounding punctuation by splitting and processing only the word part
        match = re.match(r'(^[\W_]*)([\wÀ-Ỹà-ỹ._]*[\wÀ-Ỹà-ỹ]+)([\W_]*$)', word)
        if match:
            prefix, core_word, suffix = match.groups()
            normalized_core_word = chuan_hoa_dau_tu_tieng_viet(core_word)
            words[index] = prefix + normalized_core_word + suffix
        else: # If word doesn't match (e.g. pure punctuation or malformed), try to normalize if it's a simple word
            words[index] =  (word) 
    return " ".join(words)

def is_valid_vietnam_word(word):
    chars = list(word)
    nguyen_am_index = -1
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x != -1:
            if nguyen_am_index == -1:
                nguyen_am_index = index
            else:
                if index - nguyen_am_index != 1:
                    return False
                nguyen_am_index = index
    return True

def chuan_hoa_dau_tu_tieng_viet(word):
    if not is_valid_vietnam_word(word):
        return word
    chars = list(word)
    dau_cau = 0
    nguyen_am_index = []
    qu_or_gi = False
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x == -1: continue
        if x == 9: # u
            if index > 0 and chars[index - 1].lower() == 'q':
                chars[index] = 'u'; qu_or_gi = True
        elif x == 5: # i
            if index > 0 and chars[index - 1].lower() == 'g':
                chars[index] = 'i'; qu_or_gi = True
        if y != 0:
            dau_cau = y; chars[index] = bang_nguyen_am[x][0]
        if not qu_or_gi or index != 1: # Fix: check qu_or_gi correctly
            nguyen_am_index.append(index)

    if not nguyen_am_index: return "".join(chars)

    # Determine which vowel to place the tone mark on
    idx_to_mark = nguyen_am_index[0] # Default to the first vowel in the group
    if len(nguyen_am_index) >= 2:
        # Priority for ê, ơ, ô
        priority_vowel_found = False
        for idx_candidate in nguyen_am_index:
            x_vowel, _ = nguyen_am_to_ids.get(chars[idx_candidate], (-1,-1))
            if x_vowel in [4, 7, 8]: # ê, ô, ơ
                idx_to_mark = idx_candidate
                priority_vowel_found = True
                break
        
        if not priority_vowel_found:
            # Rules for diphthongs/triphthongs (simplified from original logic)
            # If the vowel group is at the end of the word
            if nguyen_am_index[-1] == len(chars) -1:
                # If ends with i, u, y (closed vowels/semivowels), mark the vowel before it
                x_last_vowel, _ = nguyen_am_to_ids.get(chars[nguyen_am_index[-1]], (-1,-1))
                if x_last_vowel in [5, 9, 10, 11]: # i, u, ư, y
                     idx_to_mark = nguyen_am_index[-2] if len(nguyen_am_index) > 1 else nguyen_am_index[-1]
                else: # Otherwise, mark the first vowel of the group (e.g., 'oa', 'oe')
                    idx_to_mark = nguyen_am_index[0]
            else: # Vowel group is followed by consonants (e.g., 'uyen', 'oan')
                if len(nguyen_am_index) == 3: # Triphthongs like 'uye', 'oai' -> mark middle
                    idx_to_mark = nguyen_am_index[1]
                elif len(nguyen_am_index) == 2: # Diphthongs like 'uyê', 'oa' -> mark second
                    idx_to_mark = nguyen_am_index[1]
                # else (single vowel before consonant), default (first vowel) is fine

    # Apply the tone mark
    x_target_vowel, _ = nguyen_am_to_ids.get(chars[idx_to_mark], (-1,-1))
    if x_target_vowel != -1 and dau_cau != 0:
        chars[idx_to_mark] = bang_nguyen_am[x_target_vowel][dau_cau]
    return "".join(chars)

nguyen_am_to_ids = {}
for i in range(len(bang_nguyen_am)):
    for j in range(len(bang_nguyen_am[i]) - 1):
        nguyen_am_to_ids[bang_nguyen_am[i][j]] = (i, j)

def loaddicchar():
    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split('|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split('|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic
dicchar = loaddicchar()

def normalize_unicode_summ(text): # Specific to summarization
    return unicodedata.normalize('NFC', text)

def convert_unicode_legacy_summ(txt): # Specific to summarization
    # Ensure dicchar is accessible here, e.g., passed as arg or global
    # For simplicity, assuming it's global or accessible
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)


def preprocess_text_vi(text, apply_vietnamese_tone_normalization=True): # Renamed for clarity
    if not isinstance(text, str): text = str(text)
    processed_text = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", text)
    processed_text = convert_unicode_legacy_summ(processed_text)
    processed_text = normalize_unicode_summ(processed_text)
    processed_text = re.sub(emoji_pattern, " ", processed_text)
    processed_text = re.sub(r'([a-zA-Zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ])\1+', r'\1', processed_text)
    processed_text = re.sub(r'([^a-zA-Z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s])\1+', r'\1', processed_text)
    _punctuation_chars_summ = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
    escaped_punctuation_summ = re.escape(_punctuation_chars_summ)
    processed_text = re.sub(r'(?<=[^\W\d_])([' + escaped_punctuation_summ + r'])', r' \1', processed_text)
    processed_text = re.sub(r'([' + escaped_punctuation_summ + r'])(?=[^\W\d_])', r'\1 ', processed_text)
    processed_text = re.sub(r"([" + escaped_punctuation_summ + r"])\1+", r"\1", processed_text)
    if apply_vietnamese_tone_normalization:
        processed_text = chuan_hoa_dau_cau_tieng_viet(processed_text)
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    return processed_text

def preprocess_text_en(text):
    text = re.sub(r"http\S+|@\S+|#\S+", "", text) # remove URLs, mentions, hashtags
    # For nlptown model, usually just lowercasing is fine, it's multilingual.
    return text.lower()

def analyse_sentiment(text):
    if not text.strip():
        return "Vui lòng nhập văn bản.", None, None, "unknown"
    
    detected_lang = 'unknown'
    try:
        detected_lang = detect(text)
        st.info(f"Ngôn ngữ phát hiện được: {detected_lang.upper()}")
    except LangDetectException:
        st.warning("Không thể phát hiện ngôn ngữ. Mặc định xử lý bằng tiếng Anh.")
        detected_lang = 'en' # Fallback

    if detected_lang == 'vi':
        processed_text = preprocess_text_vi(text)
        tokenizer = SA_TOKENIZER_VI
        model = SA_MODEL_VI
        labels = SA_LABELS_VI
    else: # Default to English if not 'vi' or if detection failed
        processed_text = preprocess_text_en(text)
        tokenizer = SA_TOKENIZER_EN
        model = SA_MODEL_EN
        labels = SA_LABELS_EN
        if detected_lang != 'en': # If it was an unsupported lang, inform user
            st.info(f"Ngôn ngữ '{detected_lang}' chưa được hỗ trợ đầy đủ cho Phân tích Cảm xúc. Sử dụng mô hình tiếng Anh.")
            detected_lang = 'en' # Set to 'en' for label mapping

    inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    scores = outputs.logits.softmax(dim=1).squeeze()
    prediction_idx = torch.argmax(scores).item()
    
    # Ensure prediction_idx is valid for the selected labels
    if prediction_idx >= len(labels):
        # This case might happen if a model output (e.g. 5 classes) is used with fewer labels (e.g. 3 classes)
        # Or if nlptown (5 classes) is used but we only defined 3 for VI by mistake.
        # For simplicity, cap it or handle error. Here, we'll warn and pick the most likely available label.
        st.warning(f"Chỉ số dự đoán ({prediction_idx}) nằm ngoài phạm vi nhãn ({len(labels)}). Có thể có sự không khớp giữa mô hình và nhãn.")
        # Fallback: if VI model (3 labels) gave an index > 2, it's an issue with model or labels.
        # If EN model (5 labels) is used, it's fine.
        if detected_lang == 'vi' and prediction_idx >= len(SA_LABELS_VI):
             predicted_label = "Lỗi nhãn (VI)" # Or pick a default like Neutral
        elif detected_lang == 'en' and prediction_idx >= len(SA_LABELS_EN):
             predicted_label = "Label Error (EN)"
        else: # Should generally not happen if labels match model output size
             predicted_label = labels.get(prediction_idx, "Lỗi") 
    else:
        predicted_label = labels[prediction_idx]
        
    confidence = scores[prediction_idx].item()
    all_scores = {labels[i]: scores[i].item() for i in range(len(scores)) if i < len(labels)}
    return predicted_label, confidence, all_scores, detected_lang

# --- Summarization ---
def summarize_text(text, ratio=0.4):
    if not text or not text.strip():
        return "Vui lòng nhập văn bản.", "unknown"

    detected_lang = 'unknown'
    try:
        detected_lang = detect(text)
        # st.info(f"Summarization - Detected language: {detected_lang.upper()}") # Can be verbose
    except LangDetectException:
        st.warning("Không thể phát hiện ngôn ngữ cho tóm tắt. Mặc định xử lý bằng tiếng Anh.")
        detected_lang = 'en' # Fallback

    if detected_lang == 'vi':
        # Using VietAI/vit5-base-vietnews-summarization logic
        MAX_INPUT_LENGTH_SUMM = 1024
        DEFAULT_MIN_SUM_LEN_TOKENS = 100
        DEFAULT_MAX_SUM_LEN_TOKENS = 500
        ABS_MIN_SUM_LEN_TOKENS = 100 # User changed this
        ABS_MAX_SUM_LEN_TOKENS = 500 # User changed this
        NUM_BEAMS_SUMM = 4
        LENGTH_PENALTY_SUMM = 1.5
        NO_REPEAT_NGRAM_SIZE_SUMM = 3
        EARLY_STOPPING_SUMM = True
        AVG_TOKENS_PER_SENTENCE_HEURISTIC_SUMM = 25

        if SUMM_MODEL_VI is None or SUMM_TOKENIZER_VI is None:
            st.error("Mô hình tóm tắt Tiếng Việt hoặc tokenizer chưa được tải.")
            return "Lỗi: Mô hình tóm tắt Tiếng Việt chưa được tải.", 'vi'

        try:
            processed_text = preprocess_text_vi(text) # Use VI preprocessor
            if not processed_text or not processed_text.strip():
                return "Văn bản Tiếng Việt sau khi xử lý không chứa nội dung để tóm tắt.", 'vi'
            
            min_len_gen = DEFAULT_MIN_SUM_LEN_TOKENS
            max_len_gen = DEFAULT_MAX_SUM_LEN_TOKENS
            # ... (ratio-based length calculation logic - keep as is but use VI vars)
            if 0 < ratio <= 1.0:
                try:
                    sentences_in_doc = nltk.sent_tokenize(processed_text)
                    num_sentences_in_doc = len(sentences_in_doc)
                    if num_sentences_in_doc > 0:
                        target_num_sentences = max(1, int(num_sentences_in_doc * ratio))
                        ideal_summary_tokens = target_num_sentences * AVG_TOKENS_PER_SENTENCE_HEURISTIC_SUMM
                        calc_min_len = max(ABS_MIN_SUM_LEN_TOKENS, int(ideal_summary_tokens * 0.7))
                        calc_max_len = min(ABS_MAX_SUM_LEN_TOKENS, int(ideal_summary_tokens * 1.3) + 10)
                        min_len_gen = max(ABS_MIN_SUM_LEN_TOKENS, calc_min_len)
                        max_len_gen = min(ABS_MAX_SUM_LEN_TOKENS, calc_max_len)
                        if min_len_gen >= max_len_gen: min_len_gen = max(ABS_MIN_SUM_LEN_TOKENS, max_len_gen - 20); 
                        if min_len_gen < ABS_MIN_SUM_LEN_TOKENS: min_len_gen = ABS_MIN_SUM_LEN_TOKENS
                        if max_len_gen <= min_len_gen: max_len_gen = min_len_gen + 20
                        if max_len_gen > ABS_MAX_SUM_LEN_TOKENS: max_len_gen = ABS_MAX_SUM_LEN_TOKENS
                    if min_len_gen >= max_len_gen: min_len_gen = DEFAULT_MIN_SUM_LEN_TOKENS; max_len_gen = DEFAULT_MAX_SUM_LEN_TOKENS
                    if min_len_gen >= max_len_gen: min_len_gen = max(10, max_len_gen // 2)
                except Exception as e:
                    st.warning(f"Lỗi khi tính toán độ dài tóm tắt Tiếng Việt ({e}). Sử dụng độ dài mặc định.")
            min_len_gen = min(min_len_gen, MAX_INPUT_LENGTH_SUMM // 2, ABS_MAX_SUM_LEN_TOKENS -10)
            min_len_gen = max(min_len_gen, ABS_MIN_SUM_LEN_TOKENS)
            max_len_gen = max(max_len_gen, min_len_gen + 10)
            max_len_gen = min(max_len_gen, ABS_MAX_SUM_LEN_TOKENS)

            encoding = SUMM_TOKENIZER_VI(processed_text, return_tensors="pt", max_length=MAX_INPUT_LENGTH_SUMM, truncation=True, padding="longest").to(device)
            if encoding.input_ids.size(1) == 0: return "Văn bản Tiếng Việt sau khi xử lý không chứa token để tóm tắt.", 'vi'
            with torch.no_grad():
                summary_ids = SUMM_MODEL_VI.generate(encoding.input_ids, attention_mask=encoding.attention_mask, max_length=max_len_gen, min_length=min_len_gen, num_beams=NUM_BEAMS_SUMM, length_penalty=LENGTH_PENALTY_SUMM, no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE_SUMM, early_stopping=EARLY_STOPPING_SUMM)
            summary = SUMM_TOKENIZER_VI.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            return summary if summary.strip() else "Không thể tóm tắt văn bản Tiếng Việt này (kết quả tóm tắt trống).", 'vi'
        except Exception as e:
            st.error(f"Lỗi trong quá trình tóm tắt Tiếng Việt: {e}")
            return "Đã xảy ra lỗi trong quá trình tóm tắt Tiếng Việt.", 'vi'
    
    else: # Default to English summarization using SummaSummarizer
        if detected_lang != 'en':
             st.info(f"Ngôn ngữ '{detected_lang}' chưa được hỗ trợ đầy đủ cho Tóm tắt. Sử dụng SummaSummarizer (tiếng Anh).")
        try:
            # For English, preprocessing can be simpler or even none if Summa handles it well
            # processed_text_en = preprocess_text_en(text) # Optional: use simple EN preprocess
            summary = summa_summarizer(text, ratio=ratio, language='english')
            return summary if summary else "Không thể tóm tắt văn bản này (Summa), có thể văn bản quá ngắn.", 'en'
        except Exception as e:
            st.error(f"Lỗi khi tóm tắt bằng SummaSummarizer: {e}")
            return "Lỗi trong quá trình tóm tắt bằng SummaSummarizer.", 'en'

# --- Word Cloud ---
def generate_vietnamese_wordcloud(text):
    if not text.strip():
        st.warning("Vui lòng nhập văn bản để tạo word cloud.")
        return None

    text = preprocess_text_vi(text)
    
    words = text.split()
    words = [word for word in words if word not in VIETNAMESE_STOPWORDS and len(word) > 1]

    if not words:
        st.warning("Không có từ nào để tạo word cloud sau khi lọc (có thể do toàn stopwords hoặc văn bản quá ngắn).")
        return None

    wordcloud_text = " ".join(words)

    font_path = None
    common_font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", # Common on Linux
        "arial.ttf", # Common on Windows (if in system path)
        "NotoSansVietnamese-Regular.ttf" # A good fallback if installed
    ]
    for fp in common_font_paths:
        try:
            # from matplotlib.font_manager import FontProperties # REMOVED LOCAL IMPORT
            # FontProperties is now globally available
            FontProperties(fname=fp)
            font_path = fp
            break
        except Exception: # Catching a more general exception for robustness here
            continue
    
    if not font_path:
        st.warning("Không tìm thấy font hỗ trợ tiếng Việt (DejaVu Sans, Arial, Noto Sans Vietnamese). Word cloud có thể không hiển thị đúng.")
        # Fallback to default font, which likely won't render Vietnamese correctly
        
    try:
        wc = WordCloud(
            font_path=font_path,
            width=800,
            height=400,
            background_color='white',
            stopwords=VIETNAMESE_STOPWORDS # Already filtered, but good to have
        ).generate(wordcloud_text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        return fig
    except Exception as e:
        st.error(f"Lỗi khi tạo word cloud: {e}")
        if "Glyph" in str(e) or "font" in str(e):
             st.info("Lỗi này có thể liên quan đến font chữ. Hãy đảm bảo bạn có một font hỗ trợ tiếng Việt (ví dụ: DejaVu Sans, Noto Sans Vietnamese) và đường dẫn `font_path` là chính xác.")
        return None


# --- Streamlit App UI ---
st.title("Ứng dụng Xử lý Ngôn ngữ Tiếng Việt")

st.sidebar.header("Tùy chọn")
analysis_type = st.sidebar.selectbox(
    "Chọn loại phân tích:",
    ["Phân tích Cảm xúc (Sentiment Analysis)", "Tóm tắt Văn bản (Summarization)", "Đám mây Từ (Word Cloud)"]
)

st.header("Nhập văn bản Tiếng Việt vào đây:")
input_text = st.text_area(" ", height=200, key="input_text_area")

# Add a button to trigger analysis
analyze_button = st.button("Phân tích")

if analyze_button and input_text:
    if analysis_type == "Phân tích Cảm xúc (Sentiment Analysis)":
        st.subheader("Kết quả Phân tích Cảm xúc")
        predicted_label, confidence, all_scores, lang = analyse_sentiment(input_text) # Call new func
        if predicted_label and confidence is not None:
            st.write(f"**Nhãn cảm xúc dự đoán ({lang.upper()}):** {predicted_label}")
            st.write(f"**Độ tin cậy:** {confidence*100:.2f}%")
            
            st.write("---")
            st.write("**Điểm cho tất cả các nhãn:**")
            if all_scores:
                df_scores = pd.DataFrame(list(all_scores.items()), columns=['Nhãn', 'Điểm'])
                df_scores['Điểm'] = df_scores['Điểm'].apply(lambda x: f"{x*100:.2f}%")
                st.table(df_scores)
                
                # Simple bar chart for scores
                st.write("**Biểu đồ điểm cảm xúc:**")
                chart_data = pd.DataFrame(list(all_scores.items()), columns=['Nhãn', 'Điểm'])
                st.bar_chart(chart_data.set_index('Nhãn'))
            else:
                st.write("Không có điểm nào để hiển thị.")
        else:
            st.warning(predicted_label) # Handles "Vui lòng nhập văn bản."

    elif analysis_type == "Tóm tắt Văn bản (Summarization)":
        st.subheader("Kết quả Tóm tắt Văn bản")
        # Add a slider for summary ratio
        summary_ratio = st.slider("Tỷ lệ tóm tắt (càng nhỏ càng ngắn):", 0.1, 0.9, 0.4, 0.05) # Default 0.4 as per user change
        summary_result, lang = summarize_text(input_text, ratio=summary_ratio) # Call new func
        st.markdown(f"### Tóm tắt ({lang.upper()}, tỷ lệ {summary_ratio*100:.0f}%):")
        st.write(summary_result)

    elif analysis_type == "Đám mây Từ (Word Cloud)":
        st.subheader("Kết quả Đám mây Từ")
        wordcloud_fig = generate_vietnamese_wordcloud(input_text)
        if wordcloud_fig:
            st.pyplot(wordcloud_fig)
        else:
            st.info("Không thể tạo word cloud cho văn bản này.")

elif analyze_button and not input_text:
    st.warning("Vui lòng nhập văn bản vào ô trống.") 