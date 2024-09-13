import duckdb
import pandas as pd
from datasets import Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer
import torch
import random
import numpy as np
import io
from scipy.io import wavfile

torch.cuda.empty_cache() #clear the CUDA cache to free up unused memory

# === Step 1: Set random seed for reproducibility ===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# === Step 2: Load the DuckDB datasets ===
train_file = '/home/users/pvares/paper/data/train.duckdb'
test_file = '/home/users/pvares/paper/data/test.duckdb'
validation_file = '/home/users/pvares/paper/data/validation.duckdb'

# Connect to DuckDB and load each dataset into pandas DataFrames
conn = duckdb.connect(database=train_file, read_only=True)

# Load datasets
train_df = conn.execute("SELECT * FROM data").fetchdf()
conn.close()

conn = duckdb.connect(database=test_file, read_only=True)
test_df = conn.execute("SELECT * FROM data").fetchdf()
conn.close()

conn = duckdb.connect(database=validation_file, read_only=True)
validation_df = conn.execute("SELECT * FROM data").fetchdf()

# Ultimate Question of Life, the Universe and Everything
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

# === Step 4: Load the Whisper model and move to GPU ===
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Custom Data Collator for Whisper
class DataCollatorWhisper:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # Convert input_features to tensors if they are lists
        input_features = [torch.tensor(feature["input_features"]) if isinstance(feature["input_features"], list) else feature["input_features"] for feature in features]
        
        # Labels are already tensors, but ensure they are stacked as well
        labels = [feature["labels"] for feature in features]

        # Pad input features and labels
        batch = {
            "input_features": torch.stack(input_features),
            "labels": self.processor.tokenizer.pad({"input_ids": labels}, padding=True, return_tensors="pt").input_ids
        }

        return batch


# Use the custom data collator
data_collator = DataCollatorWhisper(processor)

# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.parallel.DistributedDataParallel(model)  # Distribute model across GPUs
else:
    print("Using a single GPU")

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Enable TensorFloat-32 (TF32) to help with memory optimization on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



# === Step 5: Set up the training arguments ===
training_args = TrainingArguments(
    output_dir="./whisper",                     # Directory for saving model and logs
    per_device_train_batch_size=1,              # Reduce batch size for training
    per_device_eval_batch_size=1,               # Reduce batch size for evaluation
    num_train_epochs=5,                         # Increased epochs for better convergence
    eval_strategy="epoch",                      # Evaluate at the end of each epoch
    logging_dir="./logs",                       # Directory for logging
    learning_rate=5e-5,                         # Learning rate
    weight_decay=0.01,                          # Weight decay for regularization
    save_total_limit=3,                         # Limit saved models to prevent space issues
    fp16=True,                                 # Disable mixed precision to prevent OOM issues
    logging_steps=50,                           # Log every 50 steps
    load_best_model_at_end=True,                # Load the best model based on validation metrics
    metric_for_best_model="eval_loss",          # Metric to monitor for best model
    greater_is_better=False,                    # Indicate that lower loss is better
    save_strategy="epoch",                      # Save model at the end of each epoch
    eval_steps=200,                             # Number of steps between evaluations
    seed=42,                                    # Random seed for reproducibility
    dataloader_num_workers=4,                   # Number of workers for data loading
    gradient_accumulation_steps=16,              # Accumulate gradients over steps to simulate larger batch size
    lr_scheduler_type="linear",                 # Learning rate scheduler type
    logging_first_step=True,                    # Log the first step for monitoring
    disable_tqdm=False,                         # Enable tqdm progress bar
    remove_unused_columns=False                 # Keep all columns
)



# === Step 6: Set up the Trainer and start training ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator,  # Use custom data collator
    tokenizer=processor.tokenizer  # Use tokenizer for Whisper
)

# Start training
trainer.train()

# === Step 7: Evaluate the model on the validation dataset ===
# This is done during training after each epoch, using the validation set
validation_results = trainer.evaluate(eval_dataset=validation_dataset)
print("Validation Evaluation results:", validation_results)

# === Step 8: Save the fine-tuned model ===
model.save_pretrained("./whisper")
processor.save_pretrained("./whisper")

# === Step 9: Evaluate the model on the test dataset ===
# After training, this evaluates the model on the test set (unseen data)
test_results = trainer.evaluate(eval_dataset=test_dataset)
print("Test Evaluation results:", test_results)