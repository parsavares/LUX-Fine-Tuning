import duckdb
import io
import os
import scipy.io.wavfile as wavfile

# Input: List of indexes you want to extract audio for
input_indexes = [445, 628, 642, 1092]  # Example of specific row indexes to select

# Directory to save audio files
output_dir = "audio_short"
os.makedirs(output_dir, exist_ok=True)

# Connect to the DuckDB database
conn = duckdb.connect('/home/users/pvares/paper/data/train.duckdb')

# Create a query to select the specific rows based on the provided indexes
# Assuming the table has an 'id' column or a sequential index to filter by
query = f"SELECT audio.bytes, audio.path FROM data WHERE __hf_index_id IN ({','.join(map(str, input_indexes))})"

# Fetch the data for the specific indexes
rows = conn.execute(query).fetchall()

# Process and save the filtered audio data
for row in rows:
    audio_bytes = row[0]  # The audio bytes
    file_path = row[1]  # The path of the audio
    
    # Convert bytes to a BytesIO object
    audio_io = io.BytesIO(audio_bytes)
    
    # Read the WAV data
    sampling_rate, audio_data = wavfile.read(audio_io)
    
    # Save the audio file
    output_filename = os.path.basename(file_path)  # Extract filename from the path
    output_path = os.path.join(output_dir, output_filename)  # Save in the audio_sample directory
    
    wavfile.write(output_path, sampling_rate, audio_data)
    print(f"Audio saved to {output_path}")

# Close the database connection
conn.close()
