import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModel, # Used in your load_model if model is not for sequence classification initially
    AutoModelForSequenceClassification,
    # AdamW,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW  # Import AdamW from torch.optim
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, classification_report
import re
import os
import argparse
from tqdm import tqdm # For the training loop progress bar
import emoji
import unicodedata
from underthesea import word_tokenize
from torch.utils.tensorboard import SummaryWriter # For TensorBoard
from datetime import datetime
# import mlflow # For MLflow
# import mlflow.pytorch # For MLflow PyTorch specific logging
import json # For saving args

# Global variables that will be set by args or defaults
PUNCS = '''!→()-[]{};:'"\\,<>?@#$%^&*_~'''
_stopwords_file_path = '/data/elo/khanglg/FreeTxt-Flask/vietnamese-stopwords.txt' 
try:
    with open(_stopwords_file_path, 'r', encoding='utf-8') as f:
        stopwords_list_default = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    print(f"Warning: Stopwords file not found at {_stopwords_file_path}. Stopword removal will be limited.")

vi_stopwords_global = set(stopwords_list_default)

device_global = torch.device("cpu")
_teencode_file_path = './teencode.txt' # As specified in prompt context
try:
    teencode_df = pd.read_csv(_teencode_file_path, names=['teencode', 'map'], sep='\t', header=None)
    teencode_map_default = pd.Series(teencode_df['map'].values, index=teencode_df['teencode']).to_dict()
except FileNotFoundError:
    teencode_map_default = {}

# --- Helper Functions (Adapted from your notebook) ---
def normalize_unicode_script(text): # Renamed to avoid conflict if importing notebook
    return unicodedata.normalize('NFC', text)

#Old preprocess_text_script function
# def preprocess_text_script(text):
#     """
#     Preprocesses text for sentiment analysis for the script.
#     Uses global vi_stopwords_global and PUNCS.
#     """
#     text = str(text)
#     text = re.sub(r"http\\S+|@\\S+|#\\S+", "", text)
#     text = re.sub(f"[{re.escape(''.join(PUNCS))}]", "", text.lower())
#     text = " ".join(word for word in text.split() if word not in vi_stopwords_global)
#     return text

#New preprocess_text_script function
# --- Vietnamese Character Processing Data (from Vi_preprocessing.ipynb) ---
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
bang_ky_tu_dau = ['', 'f', 's', 'r', 'x', 'j'] # Included for completeness, though not directly used by copied functions below

nguyen_am_to_ids = {}
for i in range(len(bang_nguyen_am)):
    for j in range(len(bang_nguyen_am[i]) - 1):
        nguyen_am_to_ids[bang_nguyen_am[i][j]] = (i, j)

# --- Legacy Unicode Conversion Data (from Vi_preprocessing.ipynb) ---
def loaddicchar():
    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split('|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split('|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic
dicchar = loaddicchar()

# --- Emoji Pattern (from Vi_preprocessing.ipynb) ---
# Ensure 're' is imported in the script (it is)
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

# --- Helper Functions (from Vi_preprocessing.ipynb, adapted for script) ---
def convert_unicode_legacy(txt): # Renamed to avoid clash if user defines convert_unicode
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

def is_valid_vietnam_word(word):
    # This function is part of chuan_hoa_dau_tu_tieng_viet logic
    # It checks if a word has a valid Vietnamese vowel structure.
    chars = list(word)
    nguyen_am_index = -1
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x != -1: # If char is a vowel
            if nguyen_am_index == -1: # First vowel found
                nguyen_am_index = index
            else:
                # Subsequent vowels must be adjacent to form a valid group
                if index - nguyen_am_index != 1:
                    return False # Non-adjacent vowels
                nguyen_am_index = index
    return True

def chuan_hoa_dau_tu_tieng_viet(word):
    # Normalizes tone marks for a single Vietnamese word.
    if not is_valid_vietnam_word(word):
        return word

    chars = list(word)
    dau_cau = 0 # Stores the tone mark type (0-5)
    nguyen_am_index = [] # Indices of vowels in the word
    qu_or_gi = False # Flag for 'qu' or 'gi' cases

    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1)) # x: vowel type, y: tone mark type
        if x == -1: continue # Not a vowel

        # Handle 'qu' and 'gi' cases (u/i are part of consonant)
        if x == 9: # Vowel 'u'
            if index > 0 and chars[index - 1].lower() == 'q':
                chars[index] = 'u' # Keep 'u' as is, part of 'qu'
                qu_or_gi = True
        elif x == 5: # Vowel 'i'
            if index > 0 and chars[index - 1].lower() == 'g':
                chars[index] = 'i' # Keep 'i' as is, part of 'gi'
                qu_or_gi = True
        
        if y != 0: # If there's a tone mark on this vowel
            dau_cau = y # Store the tone mark
            chars[index] = bang_nguyen_am[x][0] # Convert vowel to its base form (no tone)

        if not qu_or_gi or index != 1: # If not part of 'qu'/'gi' or not the u/i in qu/gi
            nguyen_am_index.append(index)

    if not nguyen_am_index: return "".join(chars) # No vowels to mark (e.g. 'q', 'g')

    # Determine which vowel to place the tone mark on
    idx_to_mark = nguyen_am_index[0] # Default to the first vowel in the group
    if len(nguyen_am_index) >= 2: # If multiple vowels (diphthong/triphthong)
        priority_vowel_found = False
        for idx_candidate in nguyen_am_index:
            x_vowel, _ = nguyen_am_to_ids.get(chars[idx_candidate], (-1,-1))
            if x_vowel in [4, 7, 8]: # Priority for ê (4), ô (7), ơ (8)
                idx_to_mark = idx_candidate
                priority_vowel_found = True
                break
        
        if not priority_vowel_found:
            if nguyen_am_index[-1] == len(chars) -1: # Vowel group is at the end of the word
                x_last_vowel, _ = nguyen_am_to_ids.get(chars[nguyen_am_index[-1]], (-1,-1))
                if x_last_vowel in [5, 9, 10, 11]: # Ends with i(5), u(9), ư(10), y(11)
                     idx_to_mark = nguyen_am_index[-2] if len(nguyen_am_index) > 1 else nguyen_am_index[-1]
                else: # e.g., 'oa', 'oe' -> mark the first vowel of the group
                    idx_to_mark = nguyen_am_index[0]
            else: # Vowel group is followed by consonants (e.g., 'uyen', 'oan')
                if len(nguyen_am_index) == 3: # Triphthongs like 'uye', 'oai' -> mark middle
                    idx_to_mark = nguyen_am_index[1]
                elif len(nguyen_am_index) == 2: # Diphthongs like 'uyê', 'oa' -> mark second
                    idx_to_mark = nguyen_am_index[1]
    
    # Apply the stored tone mark (dau_cau) to the determined vowel (chars[idx_to_mark])
    x_target_vowel, _ = nguyen_am_to_ids.get(chars[idx_to_mark], (-1,-1))
    if x_target_vowel != -1 and dau_cau != 0:
        chars[idx_to_mark] = bang_nguyen_am[x_target_vowel][dau_cau]
    return "".join(chars)

def chuan_hoa_dau_cau_tieng_viet(sentence):
    # Normalizes tone marks for an entire sentence.
    words = sentence.split()
    for index, word in enumerate(words):
        # Preserve surrounding punctuation by splitting and processing only the word part
        match = re.match(r'(^[\W_]*)([\wÀ-Ỹà-ỹ._]*[\wÀ-Ỹà-ỹ]+)([\W_]*$)', word, re.UNICODE)
        if match:
            prefix, core_word, suffix = match.groups()
            normalized_core_word = chuan_hoa_dau_tu_tieng_viet(core_word)
            words[index] = prefix + normalized_core_word + suffix
        else: # If word doesn't match (e.g. pure punctuation or malformed), try to normalize if it's a simple word
            words[index] = chuan_hoa_dau_tu_tieng_viet(word) 
    return " ".join(words)

# --- New Preprocessing Function ---
def preprocess_text_script(
    text,
    use_teencode=True, 
    custom_teencode_map=None,
    use_stopwords=False, 
    remove_all_punctuation=True 
    ):
    """
    Comprehensive Vietnamese text preprocessing for the script.
    Adapted from Vi_preprocessing.ipynb.
    Uses global vi_stopwords_global (must be populated).
    """
    if not isinstance(text, str):
        text = str(text)

    # Use global stopwords from the script
    current_stopwords_list = vi_stopwords_global 

    # 1. Lowercase
    processed_text = text.lower()

    # 2. Remove URLs, mentions, hashtags (using improved regex from notebook)
    processed_text = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", processed_text)

    # 3. Legacy Unicode conversion (e.g., Windows-1258 to Unicode)
    processed_text = convert_unicode_legacy(processed_text)

    # 4. Standard Unicode Normalization (NFC)
    # Uses normalize_unicode_script function already defined in finetuneSA.py
    processed_text = normalize_unicode_script(processed_text) 

    # 5. Remove Emojis (replace with space to avoid merging words)
    processed_text = re.sub(emoji_pattern, " ", processed_text)

    # 6. Reduce repeated alphabetic characters (handles Vietnamese characters)
    processed_text = re.sub(r'([a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ])\1+', r'\1', processed_text, flags=re.IGNORECASE)


    # 7. Reduce repeated special characters (non-alphanumeric, non-whitespace, non-Vietnamese)
    processed_text = re.sub(r'([^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s])\1+', r'\1', processed_text)


    # 8. Normalize punctuation spacing
    # Using a local definition of punctuation characters to avoid direct import string dependency here
    _punctuation_chars = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    escaped_punctuation = re.escape(_punctuation_chars)
    processed_text = re.sub(r"(\w)\s*([" + escaped_punctuation + r"])\s*(\w)", r"\1 \2 \3", processed_text)
    processed_text = re.sub(r"(\w)\s*([" + escaped_punctuation + r"])", r"\1 \2", processed_text)
    processed_text = re.sub(r"([" + escaped_punctuation + r"])\s*(\w)", r"\1 \2", processed_text)

    # 9. Reduce repeated punctuation characters (e.g., "!!!" to "!")
    processed_text = re.sub(r"([" + escaped_punctuation + r"])\1+", r"\1", processed_text)

    # 10. Vietnamese tone mark normalization
    processed_text = chuan_hoa_dau_cau_tieng_viet(processed_text)

    # 11. Remove all punctuation (optional)
    if remove_all_punctuation:
        translator = str.maketrans('', '', _punctuation_chars)
        processed_text = processed_text.translate(translator)

    # 12. Final whitespace cleanup (multiple spaces to single, strip ends)
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    # 13. Strip leading/trailing punctuation or space robustly (if not all removed)
    if not remove_all_punctuation and processed_text:
        _whitespace_chars = " \t\n\r\f\v" # Equivalent to string.whitespace
        strip_chars = _punctuation_chars + _whitespace_chars
        # Strip from the end
        while processed_text and processed_text[-1] in strip_chars:
            processed_text = processed_text[:-1]
        # Strip from the beginning
        while processed_text and processed_text[0] in strip_chars:
            processed_text = processed_text[1:]
    
    if not processed_text:
        return ""

    # 14. Tokenization using underthesea
    # underthesea.word_tokenize is imported in finetuneSA.py
    tokens = word_tokenize(processed_text, format="list") 
    current_teencode_map = custom_teencode_map if custom_teencode_map is not None else teencode_map_default


    # 15. Teencode Replacement (on tokens)
    if use_teencode and current_teencode_map: # custom_teencode_map will be None by default
        new_tokens = []
        for token in tokens:
            replacement = current_teencode_map.get(token, token)
            new_tokens.append(replacement)
        
        if any(" " in t for t in new_tokens): # If teencode introduced phrases
            temp_token_string = " ".join(new_tokens)
            tokens = word_tokenize(temp_token_string, format="list")
        else:
            tokens = new_tokens

    # 16. Stopword Removal (on tokens)
    if use_stopwords and current_stopwords_list: # current_stopwords_list is vi_stopwords_global
        tokens = [token for token in tokens if token not in current_stopwords_list and token.strip()]

    # 17. Join tokens to form the final processed string
    return " ".join(tokens)


def load_model_script(model_name, num_labels):
    """
    Loads a pretrained model and tokenizer for the script.
    Uses global device_global.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        model.to(device_global)
        return tokenizer, model
    except RuntimeError: # Handles cases where the model might be a base model
        print(f"Could not load {model_name} directly as AutoModelForSequenceClassification. Trying as base model.")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModel.from_pretrained(model_name)
        
        class SentimentClassifierScript(torch.nn.Module):
            def __init__(self, base_model_inner, num_labels_inner):
                super(SentimentClassifierScript, self).__init__()
                self.base_model = base_model_inner
                self.dropout = torch.nn.Dropout(0.1)
                self.classifier = torch.nn.Linear(base_model_inner.config.hidden_size, num_labels_inner)
            
            def forward(self, input_ids, attention_mask):
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.last_hidden_state[:, 0, :]
                pooled_output = self.dropout(pooled_output)
                logits = self.classifier(pooled_output)
                return logits
        
        model = SentimentClassifierScript(base_model, num_labels)
        model.to(device_global)
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None

def prepare_data_for_model_script(texts, labels, tokenizer, max_length, batch_size):
    """
    Converts texts and labels into PyTorch dataset and dataloaders for the script.
    Uses preprocess_text_script.
    Splits data 80/20 for train/validation.
    """
    print("Preprocessing texts for dataloaders...")
    preprocessed_texts = [preprocess_text_script(text) for text in tqdm(texts)]
    
    print("Tokenizing texts...")
    encodings = tokenizer(
        preprocessed_texts,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    dataset = TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        torch.tensor(labels, dtype=torch.long) # Ensure labels are torch.long
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Ensure reproducibility of split if a global seed is set for torch
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size
    )
    
    return train_dataloader, val_dataloader

def train_model_script(model, train_dataloader, val_dataloader, epochs, learning_rate, args):
    """
    Trains the model and evaluates after each epoch for the script.
    Uses global device_global.
    Logs training and validation loss, and validation accuracy to TensorBoard.
    """
    # TensorBoard writer setup
    # Ensure 'args' has 'output_dir_base' and 'model_name'
    # These would come from argparse in main()
    log_dir_base = args.output_dir_base 
    model_identifier = args.model_name.split('/')[-1]
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(log_dir_base, model_identifier, 'runs', current_time)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    optimizer = AdamW(model.parameters(), lr=learning_rate) # Reverted to simple optimizer
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0, # Reverted: No dynamic warmup from args
        num_training_steps=total_steps
    )
    
    best_val_accuracy = 0
    best_model_state = None

    # Reverted: No AMP Scaler here based on user's last file state

    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_dataloader, desc="Training"):
            batch_input_ids = batch[0].to(device_global)
            batch_attention_mask = batch[1].to(device_global)
            batch_labels = batch[2].to(device_global) # Assumes labels are at index 2
            
            model.zero_grad()
            
            # Reverted: No AMP autocast here
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip gradients
            optimizer.step()
            
            scheduler.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        print("Evaluating on validation set...")
        model.eval()
        # evaluate_model_script needs to return loss, accuracy, report
        avg_val_loss, val_accuracy, val_report = evaluate_model_script(model, val_dataloader)
        print(f"Validation Loss: {avg_val_loss:.4f}") # Print val loss
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Classification Report:\n{val_report}")

        # Log to TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_accuracy, epoch)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
            print(f"New best validation accuracy: {best_val_accuracy:.4f}")

    if best_model_state:
        print("\nLoading best model state based on validation accuracy.")
        model.load_state_dict(best_model_state)
    
    writer.close() # Close the TensorBoard writer
    return model

def evaluate_model_script(model, dataloader):
    """
    Evaluates the model and returns average loss, accuracy, and classification report for the script.
    Uses global device_global.
    """
    model.eval()
    predictions_list = []
    true_labels_list = []
    total_eval_loss = 0 # Initialize total loss for validation
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch_input_ids = batch[0].to(device_global)
            batch_attention_mask = batch[1].to(device_global)
            batch_labels = batch[2].to(device_global) # Assumes labels are at index 2

            # Pass labels to get loss during evaluation as well
            outputs = model(input_ids=batch_input_ids, 
                            attention_mask=batch_attention_mask,
                            labels=batch_labels) 
            loss = outputs.loss
            logits = outputs.logits
            
            total_eval_loss += loss.item() # Accumulate validation loss
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels_np = batch_labels.cpu().numpy()
            
            predictions_list.extend(preds)
            true_labels_list.extend(labels_np)
            
    accuracy = accuracy_score(true_labels_list, predictions_list)
    try:
        # Ensure target_names are appropriate if your labels are not 0, 1, 2
        # For now, assuming 0, 1, 2 map to some classes.
        # You might need to pass num_labels or a label map here to get proper target_names
        num_unique_labels = len(set(true_labels_list))
        target_names = [f"class_{i}" for i in range(num_unique_labels)]
        if num_unique_labels == 3: # Default for typical sentiment
            target_names = ['Tiêu cực', 'Trung tính', 'Tích cực']


        report = classification_report(true_labels_list, predictions_list, target_names=target_names, zero_division=0)
    except ValueError as e:
        print(f"Warning generating classification report: {e}. Using default report.")
        report = classification_report(true_labels_list, predictions_list, zero_division=0)

    avg_val_loss = total_eval_loss / len(dataloader) # Calculate average validation loss
    return avg_val_loss, accuracy, report # Return loss, accuracy, and report

# --- Main Execution ---
def main():
    global vi_stopwords_global, device_global

    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face model for Vietnamese Sentiment Analysis using a manual loop.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the Hugging Face model to fine-tune (e.g., 'vinai/phobert-base-v2').")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the training CSV dataset (must have 'content' and 'label' columns).")
    parser.add_argument("--stopwords_path", type=str, required=True, help="Path to the Vietnamese stopwords text file.")
    parser.add_argument("--output_dir_base", type=str, default="./fine_tuned_sa_manual_models", help="Base directory to save the fine-tuned model.")
    
    parser.add_argument("--num_labels", type=int, default=3, help="Number of sentiment classes.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length for the tokenizer.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (e.g., 'cuda:0', 'cpu').")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    # Setup device and seed
    device_global = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    print(f"Using device: {device_global}")

    # Load stopwords
    try:
        with open(args.stopwords_path, 'r', encoding='utf-8') as f:
            vi_stopwords_global = set([line.strip() for line in f if line.strip()])
        print(f"Successfully loaded {len(vi_stopwords_global)} Vietnamese stopwords from {args.stopwords_path}.")
    except FileNotFoundError:
        print(f"Warning: Vietnamese stopwords file not found at {args.stopwords_path}. Proceeding without custom stopwords.")
        vi_stopwords_global = set()

    # Prepare output directory
    model_identifier = args.model_name.split('/')[-1]
    output_dir = os.path.join(args.output_dir_base, model_identifier)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Fine-tuned model will be saved to: {output_dir}")

    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    try:
        df = pd.read_csv(args.dataset_path)
        if 'content' not in df.columns or 'label' not in df.columns:
            raise ValueError("Dataset CSV must contain 'content' and 'label' columns.")
        # Ensure labels are integers
        df['label'] = pd.to_numeric(df['label'], errors='coerce').dropna().astype(int)
        df.dropna(subset=['content', 'label'], inplace=True)

    except FileNotFoundError:
        print(f"ERROR: Dataset file not found at {args.dataset_path}.")
        return
    except ValueError as ve:
        print(f"ERROR: Value error in dataset: {ve}")
        return
    except Exception as e:
        print(f"Error loading or processing dataset: {e}")
        return
    
    texts = df['content'].tolist()
    labels = df['label'].tolist()

    # Validate num_labels
    unique_labels_in_data = set(labels)
    if args.num_labels != len(unique_labels_in_data):
        print(f"Warning: --num_labels is {args.num_labels}, but found {len(unique_labels_in_data)} unique labels in dataset: {sorted(list(unique_labels_in_data))}.")
        print("Ensure your labels are 0-indexed if num_labels is, for example, 3 (meaning labels 0, 1, 2).")
        # Potentially adjust args.num_labels or raise error if mismatch is critical

    # Load base model and tokenizer
    print(f"Loading base model '{args.model_name}' for fine-tuning...")
    tokenizer, model = load_model_script(args.model_name, args.num_labels)
    if not model or not tokenizer:
        print("Failed to load model or tokenizer. Exiting.")
        return

    # Prepare Dataloaders
    print("Preparing dataloaders...")
    train_dataloader, val_dataloader = prepare_data_for_model_script(
        texts, labels, tokenizer, args.max_seq_length, args.batch_size
    )

    # Train Model
    print(f"Starting fine-tuning of {args.model_name} for {args.epochs} epochs...")
    fine_tuned_model = train_model_script(
        model, train_dataloader, val_dataloader, args.epochs, args.learning_rate, args
    )
    print("Fine-tuning finished.")

    # Evaluate Model (on validation set, using the best state loaded in train_model_script)
    print("\nFinal evaluation of the fine-tuned model on the validation set:")
    # Adjust to receive loss, accuracy, report from evaluate_model_script
    _, final_accuracy, final_report = evaluate_model_script(fine_tuned_model, val_dataloader) 
    print(f"Final Validation Accuracy: {final_accuracy:.4f}")
    print("Final Validation Classification Report:")
    print(final_report)

    # Save Model
    print(f"Saving fine-tuned model and tokenizer to: {output_dir}")
    fine_tuned_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model and tokenizer saved successfully.")

    # Save evaluation report
    report_path = os.path.join(output_dir, "final_validation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Fine-tuned Model: {args.model_name}\n")
        f.write(f"Validation Accuracy: {final_accuracy:.4f}\n\n")
        f.write("Validation Classification Report:\n")
        f.write(final_report)
    print(f"Final validation report saved to: {report_path}")

if __name__ == "__main__":
    main()