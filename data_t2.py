import duckdb
import pandas as pd
from datasets import Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer
import torch
import random
import numpy as np
# === Step 2: Load the DuckDB datasets ===
train_file = '/home/users/pvares/paper/data/train.duckdb'
test_file = '/home/users/pvares/paper/data/test.duckdb'
validation_file = '/home/users/pvares/paper/data/validation.duckdb'

# Connect to DuckDB and load each dataset into pandas DataFrames
conn = duckdb.connect(database=train_file, read_only=True)

# List all tables in the database to verify table names
tables = conn.execute("SHOW TABLES").fetchall()
print("Tables in train database:", tables)

# Assuming there's only one table, load it into a DataFrame
train_df = conn.execute("SELECT * FROM " + tables[0][0]).fetchdf()
conn.close()  # Close the connection after loading

# Connect again for test and validation datasets
conn = duckdb.connect(database=test_file, read_only=True)
tables = conn.execute("SHOW TABLES").fetchall()
print("Tables in test database:", tables)
test_df = conn.execute("SELECT * FROM " + tables[0][0]).fetchdf()
conn.close()

conn = duckdb.connect(database=validation_file, read_only=True)
tables = conn.execute("SHOW TABLES").fetchall()
print("Tables in validation database:", tables)
validation_df = conn.execute("SELECT * FROM " + tables[0][0]).fetchdf()
conn.close()

print(train_df.head())  # Print the first few rows of the DataFrame
print(train_df.columns)  # Print the column names of the DataFrame
