import duckdb
import pandas as pd
from scipy.io import wavfile
import io
import numpy as np

# === Function to log output ===
class Logger:
    def __init__(self, log_file):
        self.log_file = log_file

    def write(self, message):
        with open(self.log_file, "a") as f:
            f.write(message + "\n")
        print(message)

# === Initialize logger ===
log = Logger("data_clean_log.txt")

# === Step 1: Load the datasets ===
log.write("Loading datasets...")
train_file = '/home/users/pvares/paper/data/train.duckdb'
validation_file = '/home/users/pvares/paper/data/validation.duckdb'
test_file = '/home/users/pvares/paper/data/test.duckdb'


# Load train dataset
conn = duckdb.connect(database=train_file, read_only=True)
train_df = conn.execute("SELECT * FROM data").fetchdf()
conn.close()

# Load validation dataset
conn = duckdb.connect(database=validation_file, read_only=True)
validation_df = conn.execute("SELECT * FROM data").fetchdf()
conn.close()

# Load test dataset
conn = duckdb.connect(database=test_file, read_only=True)
test_df = conn.execute("SELECT * FROM data").fetchdf()
conn.close()

# === Step 2: Inspect the structure of 'audio' column ===
log.write("Inspecting the structure of 'audio' column in training data...")
log.write(str(train_df["audio"].head()))

# === Step 3: Checking for Missing or Null Values ===
log.write("\n=== Step 1: Checking for Missing or Null Values ===")
log.write("Missing values in Train Dataset:")
log.write(str(train_df.isnull().sum()))
log.write("Missing values in Validation Dataset:")
log.write(str(validation_df.isnull().sum()))
log.write("Missing values in Test Dataset:")
log.write(str(test_df.isnull().sum()))

# === Step 4: Checking Audio Data Quality ===
log.write("\n=== Step 2: Checking Audio Data Quality ===")
def check_audio_duration(df, dataset_name):
    for idx, row in df.iterrows():
        audio_bytes = row["audio"]["bytes"]  # Access the audio bytes
        
        # Convert bytes to a file-like object
        audio_io = io.BytesIO(audio_bytes)

        # Read the WAV file to extract sample rate and data
        try:
            sampling_rate, audio_data = wavfile.read(audio_io)
            duration = len(audio_data) / sampling_rate  # Calculate duration
            
            if duration < 1.0:  # Check if the audio is less than 1 second
                log.write(f"Warning: Audio file at index {idx} in {dataset_name} data is too short ({duration:.2f} seconds)")
        except Exception as e:
            log.write(f"Error processing audio at index {idx} in {dataset_name} data: {e}")

# Check training, validation, and test datasets for short audios
check_audio_duration(train_df, "training")
check_audio_duration(validation_df, "validation")
check_audio_duration(test_df, "test")

# === Step 5: Checking for Duplicate Rows ===
log.write("\n=== Step 3: Checking for Duplicate Rows ===")
# Drop 'audio' column for duplicate check, as it contains unhashable types
train_df_no_audio = train_df.drop(columns=['audio'])
validation_df_no_audio = validation_df.drop(columns=['audio'])
test_df_no_audio = test_df.drop(columns=['audio'])

# Check for duplicates
duplicate_rows_train = train_df_no_audio[train_df_no_audio.duplicated()]
duplicate_rows_val = validation_df_no_audio[validation_df_no_audio.duplicated()]
duplicate_rows_test = test_df_no_audio[test_df_no_audio.duplicated()]

log.write(f"Duplicate rows in training dataset: {len(duplicate_rows_train)}")
log.write(f"Duplicate rows in validation dataset: {len(duplicate_rows_val)}")
log.write(f"Duplicate rows in test dataset: {len(duplicate_rows_test)}")

# Summary of duplicate rows
if len(duplicate_rows_train) > 0:
    log.write("Duplicate rows in training dataset:")
    log.write(str(duplicate_rows_train))
else:
    log.write("No duplicates found in training dataset.")

if len(duplicate_rows_val) > 0:
    log.write("Duplicate rows in validation dataset:")
    log.write(str(duplicate_rows_val))
else:
    log.write("No duplicates found in validation dataset.")

if len(duplicate_rows_test) > 0:
    log.write("Duplicate rows in test dataset:")
    log.write(str(duplicate_rows_test))
else:
    log.write("No duplicates found in test dataset.")
