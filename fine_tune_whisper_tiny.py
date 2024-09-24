import duckdb
import pandas as pd
from datasets import Dataset, load_from_disk
from evaluate import load
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, GenerationConfig, TrainerCallback, EarlyStoppingCallback, WhisperTokenizer
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


torch.cuda.empty_cache()  # Clear the CUDA cache to free up unused memory

# === Step 1: Set random seed for reproducibility & Visualizations===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

class TrainingVisualizationCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.train_losses = []
        self.eval_losses = []
        self.eval_wers = []
        self.learning_rates = []
        self.steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Capture the relevant metrics during training
        if "loss" in logs:
            self.train_losses.append((state.global_step, logs["loss"]))
        if "eval_loss" in logs:
            self.eval_losses.append((state.global_step, logs["eval_loss"]))
        if "eval_wer" in logs:
            self.eval_wers.append((state.global_step, logs["eval_wer"]))
        if "eval_ground_truths_wer" in logs:  # New metric
            self.eval_wers.append((state.global_step, logs["eval_ground_truths_wer"]))  # Logging ground truths WER
        if "learning_rate" in logs:
            self.learning_rates.append((state.global_step, logs["learning_rate"]))
        self.steps.append(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        # At the end of training, generate the plots
        self.plot_metrics()

    def plot_metrics(self):
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Plot loss over time
        self._plot(self.train_losses, "Training Loss", "Loss", "train_loss.png")
        self._plot(self.eval_losses, "Evaluation Loss", "Loss", "eval_loss.png")
        self._plot(self.eval_wers, "Evaluation WER", "WER", "eval_wer.png")
        self._plot(self.learning_rates, "Learning Rate", "LR", "learning_rate.png")
        self._plot(self.eval_wers, "Ground Truths WER", "Ground Truths WER", "ground_truths_wer.png")  

    def _plot(self, values, title, ylabel, filename):
        if len(values) == 0:
            return
        steps, vals = zip(*values)
        plt.figure()
        plt.plot(steps, vals)
        plt.title(title)
        plt.xlabel("Steps")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.savefig(self.output_dir / filename)
        plt.close()

# Add the callback to the trainer
visualization_callback = TrainingVisualizationCallback(output_dir="./tiny/whisper-lux-visuals")


# === Step 2: Load the DuckDB datasets or preprocessed datasets ===
train_file = '/home/users/pvares/paper/data/train.duckdb'
test_file = '/home/users/pvares/paper/data/test.duckdb'
validation_file = '/home/users/pvares/paper/data/validation.duckdb'

# Paths to save preprocessed datasets
preprocessed_train_path = "./preprocessed_train"
preprocessed_val_path = "./preprocessed_validation"
preprocessed_test_path = "./preprocessed_test"

# Load Whisper tokenizer with updated special tokens
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", use_fast=False)

# Reinitialize the processor using the updated tokenizer
processor = WhisperProcessor(tokenizer=tokenizer, feature_extractor=WhisperProcessor.from_pretrained("openai/whisper-tiny").feature_extractor)

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

'''
TEST 2
print(train_df.head())  # Check the first few rows of the training data
print(test_df.head())  # Check the first few rows of the test data
print(validation_df.head())  # Check the first few rows of the validation data


# Example of manually testing one row from the dataset
example_audio_bytes = train_df.iloc[0]["audio"]["bytes"]
audio_io = io.BytesIO(example_audio_bytes)
sampling_rate, audio_data = wavfile.read(audio_io)

# Check the properties of the audio data
print("Sampling rate:", sampling_rate)
print("Audio data shape:", audio_data.shape)
'''
'''
TEST 3

preprocessed_example = preprocess_data(train_dataset[0])
print(preprocessed_example)

'''

# === Step 4: Load the Whisper model, set the task to "transcribe", and move to GPU ===
'''
In this step, we load the Whisper model from the Hugging Face model hub using the
'WhisperForConditionalGeneration' class. The language is set to Luxembourgish, and the task
is set to "transcribe" for speech-to-text processing. Gradient checkpointing is enabled to 
optimize memory usage during training.
'''

# Load the Whisper model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

# Set the language and task to Luxembourgish speech-to-text transcription
model.generation_config.language = "luxembourgish"
model.generation_config.task = "transcribe"

# Ensure the lang_to_id is correctly set from the model's config
# model.generation_config.lang_to_id = model.config.lang_to_id

# Fix for Problem 6: Explicitly set forced_decoder_ids to None to avoid conflict
model.generation_config.forced_decoder_ids = None
model.config.forced_decoder_ids = None

# Disable use_cache to avoid conflict with gradient checkpointing
model.config.use_cache = False


# Custom Data Collator for Whisper with loss masking
'''
Here, we define a custom data collator that handles padding for input features (audio)
and labels (transcriptions), ensuring uniform batch lengths. It also replaces padding
tokens in the labels with -100 to ignore them during loss calculation and trims the 
beginning-of-sequence (BOS) token if needed.
'''

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

# Check if multiple GPUs are available
'''
This section checks whether multiple GPUs are available in the HPC environment. If more than one GPU 
is found, the model is wrapped in 'DistributedDataParallel' to allow distributed training across 
multiple GPUs. Otherwise, it will run on a single GPU.
'''
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)  # Change to DataParallel
else:
    print("Using a single GPU")

# Move the model to the appropriate device
'''
Finally, we move the model to the appropriate device (GPU or CPU). On the HPC cluster, this will 
ensure the model is loaded onto a GPU if one is available.
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Confirm the model is on the GPU
print(f"Model is on device: {next(model.parameters()).device}")

'''
Test 4

import psutil

# === GPU Information ===
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"\n=== GPU {i} Info ===")
    print(f"Name: {torch.cuda.get_device_name(i)}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(i)/1024**2:.2f} MB")
    print(f"Memory Cached: {torch.cuda.memory_reserved(i)/1024**2:.2f} MB")
    print(f"Compute Capability: {torch.cuda.get_device_capability(i)}")
    print(f"Device Properties: {torch.cuda.get_device_properties(i)}")

# === CPU Information ===
print(f"\nNumber of CPU cores: {psutil.cpu_count(logical=True)}")
print(f"CPU frequency: {psutil.cpu_freq().current} MHz")
print(f"Total RAM: {psutil.virtual_memory().total / 1024 ** 3:.2f} GB")
print(f"Available RAM: {psutil.virtual_memory().available / 1024 ** 3:.2f} GB")
'''



# === Step 5: Set up the training arguments ===
training_args = Seq2SeqTrainingArguments(
    output_dir="./tiny/whisper-lux",              # Output directory for logs and model checkpoints
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
    logging_dir="./tiny/logs",                    # Directory for logging
    disable_tqdm=False,                      # Keep the progress bar for interactive monitoring
    remove_unused_columns=False              # Keep all columns for flexibility in data processing
)


'''
Cosine learning rate schedules are preferred in fine-tuning tasks, especially for large models like Whisper, because:

1. Preventing Overfitting: Smoothly decays the learning rate, allowing careful weight adjustments in later stages, reducing the risk of overfitting.
2. Improved Exploration: Keeps learning rates higher for longer, allowing better exploration of solutions early on in training.
3. Smooth Convergence: Reduces the likelihood of instability by decaying the learning rate gradually.
4. Better Final Performance: A very small learning rate towards the end helps the model fine-tune without drastic changes, improving final accuracy.
5. Handling Long Training: Cosine schedules work well for long training runs, preventing premature convergence.

This helps Whisper fine-tune speech patterns effectively.
'''


'''
TEST 5

# Print the full training arguments to verify setup
print(training_args)

'''
# === Step 6: Define the WER metric ===

# Load the Word Error Rate (WER) metric from the Hugging Face datasets library
wer_metric = load("wer")

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
'''
This script preprocesses the datasets and trains a Whisper model for Luxembourgish transcription.
It uses two Word Error Rate (WER) metrics:
1. Standard WER: Case-sensitive and punctuation-sensitive.
2. Ground Truths WER: Case-insensitive, punctuation-insensitive, and normalizes spaces.
'''

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

# === Step 7: Set up the trainer ===

# Instantiate the Seq2SeqTrainer with the defined arguments
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[visualization_callback, EarlyStoppingCallback(early_stopping_patience=15, early_stopping_threshold=0.0005)]  # Include early stopping
)

# === Step 8: Start (or resume) training the model ===
print("Starting the training.")
#trainer.train(resume_from_checkpoint="./tiny/whisper-lux/checkpoint-5000")
trainer.train()

# === Step 9: Evaluate the model ===
# Evaluate the model on the validation set and print WER
eval_results = trainer.evaluate()

print(f"Evaluation Results: {eval_results}")

# === Step 10: Save the trained model and processor ===
model.save_pretrained("./tiny/whisper-lux-final")
processor.save_pretrained("./tiny/whisper-lux-final")
# Save the generation configuration explicitly
model.generation_config.save_pretrained("./tiny/whisper-lux-final")