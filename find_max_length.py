import duckdb
import pandas as pd
from datasets import Dataset
from transformers import WhisperProcessor
import os

# === Step 1: Load the DuckDB datasets ===
train_file = '/home/users/pvares/paper/data/train.duckdb'
test_file = '/home/users/pvares/paper/data/test.duckdb'
validation_file = '/home/users/pvares/paper/data/validation.duckdb'

# Connect to DuckDB and load the dataset into pandas DataFrame (using train data for this example)
conn = duckdb.connect(database=train_file, read_only=True)
train_df = conn.execute("SELECT * FROM data").fetchdf()
conn.close()

# Convert pandas DataFrame to Hugging Face dataset
train_dataset = Dataset.from_pandas(train_df)

# === Step 2: Load the Whisper processor ===
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

# === Step 3: Calculate the max token length based on transcriptions ===
def find_max_length(dataset):
    max_length = 0
    for example in dataset:
        transcription = example["transcription"]
        # Tokenize the transcription
        input_ids = processor.tokenizer(transcription, return_tensors="pt").input_ids
        # Update max_length if this transcription is longer
        max_length = max(max_length, input_ids.shape[1])
    return max_length

# Find the maximum token length for the train dataset
max_token_length = find_max_length(train_dataset)

# === Step 4: Output the result ===
print(f"Maximum token length in the training dataset: {max_token_length}")
