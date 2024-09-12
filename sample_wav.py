import duckdb
import io
import os
import scipy.io.wavfile as wavfile

'''

# Connect to the DuckDB database
conn = duckdb.connect('/home/users/pvares/paper/data/train.duckdb')

# Check the structure of the data table
query = "DESCRIBE data"
columns = conn.execute(query).fetchall()
print("Columns in the data table:", columns)

'''

# Directory to save audio files
output_dir = "audio_sample"
os.makedirs(output_dir, exist_ok=True)

# Connect to the DuckDB database
conn = duckdb.connect('/home/users/pvares/paper/data/train.duckdb')

# Query specific audio data
query = "SELECT audio.bytes, audio.path FROM data LIMIT 5"  # Adjust this query as needed

# Fetch the data
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