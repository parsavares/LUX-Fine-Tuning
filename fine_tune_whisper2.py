import duckdb
import pandas as pd
from datasets import Dataset, load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer
import torch
import random
import numpy as np
import io
from scipy.io import wavfile
import os

torch.cuda.empty_cache()  # Clear the CUDA cache to free up unused memory

# === Step 1: Set random seed for reproducibility ===
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
    # Load the Whisper processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-large")

    # Updated preprocessing function to add padding
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

        # Process the transcription to obtain decoder_input_ids
        transcription = examples["transcription"]
        labels = processor.tokenizer(transcription, return_tensors="pt", padding=True, truncation=True).input_ids

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
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")

# Set the language and task to Luxembourgish speech-to-text transcription
model.generation_config.language = "luxembourgish"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None  # Optional, ensure no forced tokens are added

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

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
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Handle labels (transcriptions)
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

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
    output_dir="./whisper-lux",
    save_steps=1000,  # Save model every 1000 steps
    save_total_limit=3,  # Keep a maximum of 3 checkpoints to prevent using too much space
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

'''
TEST 5

# Print the full training arguments to verify setup
print(training_args)

'''
# === Step 6: Define the WER metric ===

# Load the Word Error Rate (WER) metric from the Hugging Face datasets library
wer_metric = load_metric("wer")

# Function to compute WER for evaluation
def compute_metrics(pred):
    # Get predicted and reference ids
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Decode predicted and reference transcriptions
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id  # Ignore padding
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER score
    wer_score = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer_score}

# === Step 7: Set up the trainer ===
# Instantiate the Seq2SeqTrainer with the defined arguments
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# === Step 8: Start (or resume) training the model ===

# Define the checkpoint path based on the training output directory
checkpoint_dir = "./whisper-lux"

# Check if there's a checkpoint available
if os.path.exists(checkpoint_dir):
    print(f"Resuming training from checkpoint: {checkpoint_dir}")
    trainer.train(resume_from_checkpoint=checkpoint_dir)
else:
    print("No checkpoint found. Starting training from scratch.")
    trainer.train()
# === Step 9: Evaluate the model ===
# Evaluate the model on the validation set and print WER
eval_results = trainer.evaluate()

print(f"Evaluation Results: {eval_results}")

# === Step 10: Save the trained model and processor ===
model.save_pretrained("./whisper-lux-final")
processor.save_pretrained("./whisper-lux-final")