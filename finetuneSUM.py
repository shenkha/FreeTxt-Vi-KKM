# finetune_summarization.py

import argparse
import os
import json
import re
import unicodedata
from datetime import datetime

import torch
import nltk
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
import evaluate 

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


def preprocess_text_for_abstractive_summarization(text, apply_vietnamese_tone_normalization=True):
    if not isinstance(text, str): text = str(text)
    processed_text = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", text)
    processed_text = convert_unicode_legacy_summ(processed_text)
    processed_text = normalize_unicode_summ(processed_text)
    processed_text = re.sub(emoji_pattern, " ", processed_text)
    processed_text = re.sub(r'([a-zA-Zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ])\1+', r'\1', processed_text)
    processed_text = re.sub(r'([^a-zA-Z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s])\1+', r'\1', processed_text)
    _punctuation_chars_summ = '!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    escaped_punctuation_summ = re.escape(_punctuation_chars_summ)
    processed_text = re.sub(r'(?<=[^\W\d_])([' + escaped_punctuation_summ + r'])', r' \1', processed_text)
    processed_text = re.sub(r'([' + escaped_punctuation_summ + r'])(?=[^\W\d_])', r'\1 ', processed_text)
    processed_text = re.sub(r"([" + escaped_punctuation_summ + r"])\1+", r"\1", processed_text)
    if apply_vietnamese_tone_normalization: # Control tone normalization
        processed_text = chuan_hoa_dau_cau_tieng_viet(processed_text)
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    return processed_text

# --- Main Fine-tuning Logic ---
def main(args):
    print(f"Starting fine-tuning with arguments: {args}")

    # --- 1. Load Tokenizer and Model ---
    print(f"Loading tokenizer and model from: {args.model_name_or_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        return

    # --- 2. Load and Prepare Dataset ---
    print(f"Loading dataset from: {args.dataset_name_or_path}")
    raw_datasets = DatasetDict()
    try:
        if os.path.exists(args.dataset_name_or_path):
            file_ext = os.path.splitext(args.dataset_name_or_path)[1].lower()
            if file_ext == ".csv":
                raw_datasets['train'] = load_dataset('csv', data_files={'train': args.dataset_name_or_path})['train']
            elif file_ext in [".json", ".jsonl"]:
                 raw_datasets['train'] = load_dataset('json', data_files={'train': args.dataset_name_or_path})['train']
            else:
                raise ValueError(f"Unsupported local file format: {file_ext}. Use .csv or .json/.jsonl")
            
            # If a separate validation file is provided
            if args.validation_file and os.path.exists(args.validation_file):
                val_file_ext = os.path.splitext(args.validation_file)[1].lower()
                if val_file_ext == ".csv":
                    raw_datasets['validation'] = load_dataset('csv', data_files={'validation': args.validation_file})['validation']
                elif val_file_ext in [".json", ".jsonl"]:
                    raw_datasets['validation'] = load_dataset('json', data_files={'validation': args.validation_file})['validation']
                else:
                    print(f"Warning: Unsupported validation file format: {val_file_ext}. Ignoring.")
            elif args.validation_split_percentage > 0:
                 # Split train into train/validation
                train_test_split_data = raw_datasets['train'].train_test_split(
                    test_size=args.validation_split_percentage / 100.0, 
                    seed=args.seed
                )
                raw_datasets['train'] = train_test_split_data['train']
                raw_datasets['validation'] = train_test_split_data['test']
            else:
                print("Warning: No validation set provided or split percentage set. Evaluation will be limited.")
                # Create a tiny dummy validation set if none, Trainer might require it
                if 'validation' not in raw_datasets and 'train' in raw_datasets and len(raw_datasets['train']) > 10:
                    dummy_val = raw_datasets['train'].select(range(min(10, len(raw_datasets['train']))))
                    raw_datasets['validation'] = dummy_val
                    print("Created a small dummy validation set from training data for Trainer compatibility.")


        else: # Try loading from Hugging Face Hub
            dataset_parts = args.dataset_name_or_path.split(':') # e.g. "dataset_name:subset_name"
            dataset_name = dataset_parts[0]
            subset_name = dataset_parts[1] if len(dataset_parts) > 1 else None
            
            loaded_data = load_dataset(dataset_name, name=subset_name, trust_remote_code=args.trust_remote_code)
            
            if 'train' not in loaded_data:
                 raise ValueError(f"'train' split not found in Hub dataset {args.dataset_name_or_path}")
            raw_datasets['train'] = loaded_data['train']

            if args.dataset_val_split_name and args.dataset_val_split_name in loaded_data:
                raw_datasets['validation'] = loaded_data[args.dataset_val_split_name]
            elif args.validation_split_percentage > 0 :
                train_test_split_data = raw_datasets['train'].train_test_split(
                    test_size=args.validation_split_percentage / 100.0,
                    seed=args.seed
                )
                raw_datasets['train'] = train_test_split_data['train']
                raw_datasets['validation'] = train_test_split_data['test']
            else:
                print("Warning: No validation set specified for Hub dataset or split percentage set. Evaluation may be limited.")
                if 'validation' not in raw_datasets and 'train' in raw_datasets and len(raw_datasets['train']) > 10:
                    raw_datasets['validation'] = raw_datasets['train'].select(range(min(10, len(raw_datasets['train'])))) # Dummy
                    print("Created a small dummy validation set for Hub dataset.")


        print(f"Train dataset size: {len(raw_datasets.get('train', []))}")
        if 'validation' in raw_datasets:
            print(f"Validation dataset size: {len(raw_datasets.get('validation', []))}")

    except Exception as e:
        print(f"Error loading or splitting dataset: {e}")
        return

    # --- 3. Preprocess and Tokenize Datasets ---
    def preprocess_for_trainer(examples):
        inputs = [preprocess_text_for_abstractive_summarization(doc, args.apply_vietnamese_normalization) for doc in examples[args.document_column]]
        targets = [preprocess_text_for_abstractive_summarization(summary, args.apply_vietnamese_normalization) for summary in examples[args.summary_column]]
        
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, truncation=True, padding="longest")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=args.max_target_length, truncation=True, padding="longest")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if not raw_datasets.get('train'):
        print("No training data available. Exiting.")
        return

    print("Tokenizing datasets...")
    try:
        tokenized_datasets = raw_datasets.map(
            preprocess_for_trainer,
            batched=True,
            remove_columns=list(raw_datasets['train'].column_names) # Use columns from actual train split
        )
    except Exception as e:
        print(f"Error during tokenization: {e}")
        # Potentially print one example to see if column names are correct
        print("Example train instance keys:", raw_datasets['train'][0].keys() if len(raw_datasets['train']) > 0 else "Train set empty")
        return
        
    print("Tokenization complete.")

    # --- 4. Data Collator ---
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # --- 5. Metrics for Evaluation ---
    rouge_metric = evaluate.load("rouge")
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple): preds = preds[0]
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Post-process for ROUGE (newline between sentences)
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {key: value * 100 for key, value in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return {k: round(v, 4) for k, v in result.items()}

    # --- 6. Training Arguments ---
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{os.path.basename(args.model_name_or_path)}_{time_stamp}")
    logging_dir = os.path.join(output_dir, "logs")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        evaluation_strategy="epoch" if 'validation' in tokenized_datasets and len(tokenized_datasets['validation']) > 0 else "no",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        predict_with_generate=True,
        fp16=args.fp16 and torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True if 'validation' in tokenized_datasets and len(tokenized_datasets['validation']) > 0 else False,
        metric_for_best_model=args.metric_for_best_model if 'validation' in tokenized_datasets and len(tokenized_datasets['validation']) > 0 else None,
        greater_is_better=True, # Assuming higher ROUGE is better
        report_to=["tensorboard"] if args.use_tensorboard else ["none"],
        seed=args.seed,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
    )
    
    callbacks = []
    if args.early_stopping_patience > 0 and 'validation' in tokenized_datasets and len(tokenized_datasets['validation']) > 0 :
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    # --- 7. Initialize Trainer ---
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets.get("train"),
        eval_dataset=tokenized_datasets.get("validation") if 'validation' in tokenized_datasets and len(tokenized_datasets['validation']) > 0 else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if 'validation' in tokenized_datasets and len(tokenized_datasets['validation']) > 0 else None,
        callbacks=callbacks
    )

    # --- 8. Start Training ---
    print(f"--- Starting Fine-Tuning: Output will be in {output_dir} ---")
    try:
        if tokenized_datasets.get("train"):
            train_result = trainer.train()
            print("Fine-tuning complete!")
            trainer.save_model()
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()
            print(f"Model, tokenizer, and training state saved to {output_dir}")

            if tokenized_datasets.get("validation") and len(tokenized_datasets['validation']) > 0:
                print("\n--- Evaluating the best model on the validation set ---")
                eval_metrics = trainer.evaluate()
                trainer.log_metrics("eval_best", eval_metrics)
                trainer.save_metrics("eval_best", eval_metrics)
                print(f"Evaluation metrics for the best model: {eval_metrics}")
        else:
            print("No training data found. Skipping training.")

    except Exception as e:
        print(f"An error occurred during training or evaluation: {e}")
        import traceback
        traceback.print_exc()


    print(f"\n--- Fine-Tuning Script Finished ---")
    if args.use_tensorboard:
        print(f"To view TensorBoard logs, run: tensorboard --logdir='{logging_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Seq2Seq model for Summarization.")

    # Model and Tokenizer
    parser.add_argument("--model_name_or_path", type=str, default="VietAI/vit5-base-vietnews-summarization",
                        help="Path to pre-trained model or model identifier from Hugging Face Hub.")

    # Dataset
    parser.add_argument("--dataset_name_or_path", type=str, required=True,
                        help="Dataset identifier from Hub (e.g., 'OpenHust/vietnamese-summarization') or path to local .csv/.json file.")
    parser.add_argument("--dataset_val_split_name", type=str, default="validation",
                        help="Name of the validation split if loading from Hub (e.g., 'validation', 'test').")
    parser.add_argument("--validation_file", type=str, default=None,
                        help="Path to a separate local validation file (.csv or .json).")
    parser.add_argument("--validation_split_percentage", type=float, default=10.0,
                        help="Percentage of training data to use for validation if no validation_file or Hub validation split is provided (0 to disable).")
    parser.add_argument("--document_column", type=str, default="Document", help="Name of the column containing the document/article text.")
    parser.add_argument("--summary_column", type=str, default="Summary", help="Name of the column containing the summary text.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Set to true if dataset loading requires trusting remote code.")
    parser.add_argument("--apply_vietnamese_normalization", action="store_true", help="Apply Vietnamese-specific tone normalization.")


    # Directories
    parser.add_argument("--output_dir", type=str, default="./finetuned_summarization_models",
                        help="Directory to save the fine-tuned model, checkpoints, and logs.")

    # Training Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Initial learning rate.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning rate scheduler type (e.g., 'linear', 'cosine').")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of total training steps for linear warmup.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Limit the total number of checkpoints. Deletes older checkpoints.")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training (if GPU available).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--metric_for_best_model", type=str, default="rouge2", help="Metric to use for identifying the best model.")
    parser.add_argument("--early_stopping_patience", type=int, default=0, help="Number of epochs with no improvement after which training will be stopped. 0 to disable.")


    # Generation Hyperparameters (for predict_with_generate)
    parser.add_argument("--generation_max_length", type=int, default=256, help="Maximum length for generated summaries during evaluation.")
    parser.add_argument("--generation_num_beams", type=int, default=4, help="Number of beams for beam search during evaluation.")
    
    # Sequence Lengths
    parser.add_argument("--max_source_length", type=int, default=1024, help="Maximum input sequence length.")
    parser.add_argument("--max_target_length", type=int, default=256, help="Maximum target sequence length (summary).")

    # Logging
    parser.add_argument("--use_tensorboard", action="store_true", help="Enable TensorBoard logging.")
    
    args = parser.parse_args()
    
    # Download nltk punkt if not already present (for ROUGE metric postprocessing)
    try:
        nltk.data.find('tokenizers/punkt')
    except (nltk.downloader.DownloadError, LookupError):
        nltk.download('punkt', quiet=True)

    main(args)