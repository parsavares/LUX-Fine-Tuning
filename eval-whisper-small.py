import duckdb
import pandas as pd
from datasets import Dataset, load_from_disk
from evaluate import load
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig, TrainerCallback, EarlyStoppingCallback, WhisperTokenizer, Seq2SeqTrainingArguments
import torch
import random
import numpy as np
import io
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt
from pathlib import Path
import json
import re
from transformers import GenerationConfig


torch.cuda.empty_cache()  # Clear the CUDA cache to free up unused memory

# Set random seed for reproducibility & Visualizations===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# === Step 2: Load the DuckDB datasets or preprocessed datasets ===
train_file = '/home/users/pvares/paper/data/train.duckdb'
test_file = '/home/users/pvares/paper/data/test.duckdb'
validation_file = '/home/users/pvares/paper/data/validation.duckdb'

# Paths to save preprocessed datasets
preprocessed_train_path = "./preprocessed_train"
preprocessed_val_path = "./preprocessed_validation"
preprocessed_test_path = "./preprocessed_test"

# Load Whisper tokenizer with updated special tokens
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", use_fast=False)

# Reinitialize the processor using the updated tokenizer
processor = WhisperProcessor(tokenizer=tokenizer, feature_extractor=WhisperProcessor.from_pretrained("openai/whisper-small").feature_extractor)

# Check if preprocessed datasets exist
if os.path.exists(preprocessed_train_path) and os.path.exists(preprocessed_val_path) and os.path.exists(preprocessed_test_path):
    print("Loading preprocessed datasets from disk...")
    train_dataset = load_from_disk(preprocessed_train_path)
    validation_dataset = load_from_disk(preprocessed_val_path)
    test_dataset = load_from_disk(preprocessed_test_path)
else:
    print("Preprocessing datasets for the first time...")

    # Connect to DuckDB and load each dataset into pandas DataFrames
    conn = duckdb.connect(database=train_file, read_only=True)
    train_df = conn.execute("SELECT * FROM data").fetchdf()
    conn.close()

    conn = duckdb.connect(database=test_file, read_only=True)
    test_df = conn.execute("SELECT * FROM data").fetchdf()
    conn.close()

    conn = duckdb.connect(database=validation_file, read_only=True)
    validation_df = conn.execute("SELECT * FROM data").fetchdf()
    conn.close()

    # Convert pandas DataFrames to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    validation_dataset = Dataset.from_pandas(validation_df)

    # === Step 3: Preprocess the audio and transcription ===

    # Updated preprocessing function with padding and truncation for tokenization
    def preprocess_data(examples):
        # Access the audio bytes directly
        audio_bytes = examples["audio"]["bytes"]
    
        # Convert the bytes to a suitable format using BytesIO
        audio_io = io.BytesIO(audio_bytes)
    
        # Read the WAV file from the bytes buffer
        sampling_rate, audio_data = wavfile.read(audio_io)
    
        # Ensure audio data is in float32 format (normalized)
        audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data))
    
        # Process the audio data for the model
        inputs = processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt").input_features

        # Process the transcription to obtain decoder_input_ids with padding and truncation
        transcription = examples["transcription"]
        labels = processor.tokenizer(
            transcription, 
            return_tensors="pt", 
            padding="max_length",  # Ensure consistent length for labels
            truncation=True,       # Truncate long sequences to match the max length
            max_length=367         # Ensure the same max length as the model's generation length
        ).input_ids

        return {"input_features": inputs.squeeze(0), "labels": labels.squeeze(0)}


    # Apply preprocessing to the datasets
    train_dataset = train_dataset.map(preprocess_data)
    validation_dataset = validation_dataset.map(preprocess_data)
    test_dataset = test_dataset.map(preprocess_data)

    # Save preprocessed datasets to disk
    train_dataset.save_to_disk(preprocessed_train_path)
    validation_dataset.save_to_disk(preprocessed_val_path)
    test_dataset.save_to_disk(preprocessed_test_path)

    
# Load the trained model and processor
model = WhisperForConditionalGeneration.from_pretrained("./small/whisper-lux-final")
processor = WhisperProcessor.from_pretrained("./small/whisper-lux-final")
# Define the generation configuration separately
generation_config = GenerationConfig.from_pretrained("./small/whisper-lux-final")
model.generation_config = generation_config

# Set the language and task to Luxembourgish speech-to-text transcription
model.generation_config.language = "luxembourgish"
model.generation_config.task = "transcribe"

# Fix for Problem 6: Explicitly set forced_decoder_ids to None to avoid conflict
model.generation_config.forced_decoder_ids = None
model.config.forced_decoder_ids = None

# Disable use_cache to avoid conflict with gradient checkpointing
model.config.use_cache = False

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Handle input features (audio)
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        # Add attention mask to handle padding correctly
        batch["attention_mask"] = torch.ones(batch["input_features"].shape, dtype=torch.long)

        # Handle labels (transcriptions)
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding tokens in labels with -100 for loss masking
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Trim the BOS token if present
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        # Add the processed labels to the batch
        batch["labels"] = labels
        return batch

# Instantiate the data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)  # Change to DataParallel
else:
    print("Using a single GPU")

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Confirm the model is on the GPU
print(f"Model is on device: {next(model.parameters()).device}")

# Load the Word Error Rate (WER) metric
wer_metric = load("wer")

# Function to normalize text
def normalize_text(text: str) -> str:
    """
    Normalize text by:
    1. Lowercasing.
    2. Removing punctuation.
    3. Removing extra spaces.
    """
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to compute WER for evaluation
def compute_metrics(pred):
    # Get predicted and reference ids
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Decode predicted and reference transcriptions
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id  # Ignore padding
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER score (original version)
    wer_score = wer_metric.compute(predictions=pred_str, references=label_str)

    # Compute ground_truths WER score (case/punctuation/space-insensitive)
    normalized_pred_str = [normalize_text(p) for p in pred_str]
    normalized_label_str = [normalize_text(l) for l in label_str]
    ground_truths_wer = wer_metric.compute(predictions=normalized_pred_str, references=normalized_label_str)

    return {"wer": wer_score, "ground_truths_wer": ground_truths_wer}

# === Step 5: Set up the training arguments ===
training_args = Seq2SeqTrainingArguments(
    output_dir="./small/whisper-lux",              # Output directory for logs and model checkpoints
    save_steps=1000,                         # Save model every 500 steps for frequent checkpointing
    save_total_limit=5,                      # Keep up to 5 checkpoints to avoid space issues
    per_device_train_batch_size=4,           # Use a larger batch size (increase as much as your GPU allows)
    per_device_eval_batch_size=8,            # Keep eval batch size larger for faster evaluation
    gradient_accumulation_steps=1,           # Reduce or remove gradient accumulation for more frequent updates
    learning_rate=1e-5,                      # Use a lower learning rate to prevent overfitting and ensure smoother convergence
    warmup_steps=1000,                       # Keep warm-up steps to stabilize early training
    max_steps=20000,                         # Increase total steps to allow more training iterations
    gradient_checkpointing=True,             # Keep gradient checkpointing to save memory
    fp16=False,                              # Use full precision (`fp32`) for maximum quality
    eval_strategy="steps",                   # Evaluate periodically by steps 
    eval_steps=1000,                          # Frequent evaluation (every 500 steps)
    predict_with_generate=True,              # For seq2seq tasks, generate predictions during evaluation
    generation_max_length=367,               # Set max length for predictions (based on dataset)
                                             #=> Maximum token length in the training dataset: 347 +20 => 367 (/home/users/pvares/paper/find_max_length.py)
    logging_steps=50,                        # Log every 25 steps for detailed tracking
    weight_decay=0.01,                       # Adding weight decay to regularize large weights
    report_to=["tensorboard"],               # Log to TensorBoard for visualization
    load_best_model_at_end=True,             # Always load the best model based on evaluation metric
    metric_for_best_model="wer",             # Track Word Error Rate (WER) for best model selection
    greater_is_better=False,                 # Lower WER is better
    push_to_hub=False,                       # Disable model push to Hugging Face hub
    dataloader_num_workers=6,                # Use multiple workers for data loading to speed up the process
    lr_scheduler_type="cosine",# Use cosine learning rate scheduler for better learning rate decay
    seed=42,                                 # Ensure reproducibility with a fixed random seed
    logging_dir="./small/logs",                    # Directory for logging
    disable_tqdm=False,                      # Keep the progress bar for interactive monitoring
    remove_unused_columns=False              # Keep all columns for flexibility in data processing
)

# Initialize Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation Results: {eval_results}")
