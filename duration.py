import duckdb
import numpy as np
import io
import os
from scipy.io import wavfile

output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# File to save the output
output_file = os.path.join(output_dir, "Data_duration.txt")

# Paths to the DuckDB databases for train, test, and validation datasets
train_file = '/home/users/pvares/paper/data/train.duckdb'
test_file = '/home/users/pvares/paper/data/test.duckdb'
validation_file = '/home/users/pvares/paper/data/validation.duckdb'

# Function to calculate total duration of audio data
def calculate_total_duration(database_path):
    # Connect to the DuckDB database
    conn = duckdb.connect(database=database_path, read_only=True)
    
    # Fetch the data, assuming the table name is 'data' and audio is stored in a column 'audio'
    # Replace 'audio' and 'data' with appropriate column/table names if different
    query = "SELECT audio['bytes'] AS audio_bytes FROM data"
    df = conn.execute(query).fetchdf()
    
    conn.close()

    total_duration = 0.0  # To store the total duration in seconds
    
    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        audio_bytes = row['audio_bytes']
        
        # Convert the bytes to a BytesIO object
        audio_io = io.BytesIO(audio_bytes)
        
        # Read the audio file from the bytes object
        sampling_rate, audio_data = wavfile.read(audio_io)
        
        # Calculate the duration of the audio in seconds and add to total
        duration = len(audio_data) / sampling_rate
        total_duration += duration

    return total_duration

# Calculate durations for each dataset
train_duration = calculate_total_duration(train_file)
test_duration = calculate_total_duration(test_file)
validation_duration = calculate_total_duration(validation_file)

# Convert durations to hours for better readability
train_duration_hours = train_duration / 3600
test_duration_hours = test_duration / 3600
validation_duration_hours = validation_duration / 3600

with open(output_file, "w") as f:
    # Print out the durations
    f.write(f"Total duration of train dataset: {train_duration_hours:.2f} hours \n")
    f.write(f"Total duration of test dataset: {test_duration_hours:.2f} hours \n")
    f.write(f"Total duration of validation dataset: {validation_duration_hours:.2f} hours \n")
print(f"Output saved to {output_file}")
 