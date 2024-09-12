import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define paths to DuckDB files and output directories
db_paths = {
    'train': '/home/users/pvares/paper/data/train.duckdb',
    'validation': '/home/users/pvares/paper/data/validation.duckdb',
    'test': '/home/users/pvares/paper/data/test.duckdb'
}

output_dirs = {
    'train': 'train_data',
    'validation': 'validate_data',
    'test': 'test_data'
}

# Create directories for saving images if they don't exist
for output_dir in output_dirs.values():
    os.makedirs(output_dir, exist_ok=True)

# Define a function to load DuckDB data and perform visualizations
def process_data(db_path, output_dir):
    # Connect to the DuckDB file
    conn = duckdb.connect(db_path)

    # List all tables in the database
    tables = conn.execute("SHOW TABLES").fetchall()

    if len(tables) == 0:
        print(f"No tables found in {db_path}")
        return

    # Extract the first table name
    table_name = tables[0][0]

    # Query the data from the table
    df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()

    # Close the DuckDB connection
    conn.close()

    # Print an overview of the dataset
    print(f"Data Overview for {output_dir}:")
    print(df.head())

    # 1. Audio Duration Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['audio.duration'], bins=30, kde=True)
    plt.title("Audio Duration Distribution")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(f'{output_dir}/audio_duration_distribution.png')
    plt.close()

    # 2. Boxplot of Audio Duration
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['audio.duration'])
    plt.title("Boxplot of Audio Duration")
    plt.xlabel("Duration (seconds)")
    plt.grid(True)
    plt.savefig(f'{output_dir}/audio_duration_boxplot.png')
    plt.close()

    # 3. Scatterplot of Transcription Length vs Audio Duration
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['audio.duration'], y=df['transcription.length'])
    plt.title("Audio Duration vs Transcription Length")
    plt.xlabel("Audio Duration (seconds)")
    plt.ylabel("Transcription Length (characters)")
    plt.grid(True)
    plt.savefig(f'{output_dir}/audio_vs_transcription_scatterplot.png')
    plt.close()

    # 4. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr_matrix = df[['audio.duration', 'transcription.length']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.savefig(f'{output_dir}/correlation_heatmap.png')
    plt.close()

    # 5. Transcription Length by Audio Duration Categories
    df['duration_category'] = pd.cut(df['audio.duration'], bins=5)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='duration_category', y='transcription.length', data=df)
    plt.title("Transcription Length by Audio Duration Category")
    plt.xlabel("Audio Duration Category")
    plt.ylabel("Transcription Length (characters)")
    plt.grid(True)
    plt.savefig(f'{output_dir}/transcription_length_by_duration_category.png')
    plt.close()

    # 6. Countplot of Audio Duration Categories
    plt.figure(figsize=(10, 6))
    sns.countplot(x='duration_category', data=df)
    plt.title("Count of Audio Files by Duration Category")
    plt.xlabel("Audio Duration Category")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig(f'{output_dir}/audio_duration_countplot.png')
    plt.close()

    # 7. Distribution of Transcription Lengths
    plt.figure(figsize=(10, 6))
    sns.histplot(df['transcription.length'], bins=30, kde=True)
    plt.title("Transcription Length Distribution")
    plt.xlabel("Transcription Length (characters)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(f'{output_dir}/transcription_length_distribution.png')
    plt.close()

    # 8. Violin Plot of Transcription Length by Audio Duration Category
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='duration_category', y='transcription.length', data=df)
    plt.title("Violin Plot of Transcription Length by Audio Duration Category")
    plt.xlabel("Audio Duration Category")
    plt.ylabel("Transcription Length (characters)")
    plt.grid(True)
    plt.savefig(f'{output_dir}/transcription_length_violinplot.png')
    plt.close()

    # 9. Cumulative Distribution Plot for Audio Duration
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(df['audio.duration'])
    plt.title("Cumulative Distribution of Audio Duration")
    plt.xlabel("Audio Duration (seconds)")
    plt.ylabel("Cumulative Probability")
    plt.grid(True)
    plt.savefig(f'{output_dir}/audio_duration_cdf.png')
    plt.close()

    # 10. Pairplot of Numeric Features
    plt.figure(figsize=(10, 6))
    sns.pairplot(df[['audio.duration', 'transcription.length']])
    plt.suptitle("Pairplot of Numeric Features", y=1.02)
    plt.savefig(f'{output_dir}/numeric_features_pairplot.png')
    plt.close()

# Process each dataset
for dataset, db_path in db_paths.items():
    process_data(db_path, output_dirs[dataset])
