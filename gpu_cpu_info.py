import torch
import os
import psutil
import subprocess

# Create a directory to save the output if it doesn't exist
output_dir = "GPU_CPU"
os.makedirs(output_dir, exist_ok=True)

# File to save the output
output_file = os.path.join(output_dir, "gpu_cpu_info.txt")

# Open the file to write the output
with open(output_file, "w") as f:
    # Check if CUDA is available and print GPU info
    f.write(f"Is CUDA available? {torch.cuda.is_available()}\n")
    f.write(f"Number of GPUs: {torch.cuda.device_count()}\n")

    if torch.cuda.is_available():
        f.write(f"GPU Name: {torch.cuda.get_device_name(0)}\n")
        f.write(f"GPU Architecture: {torch.cuda.get_arch_list()}\n")

    # Get CPU info
    cpu_count = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    ram_info = psutil.virtual_memory()

    f.write(f"Number of CPU cores: {cpu_count}\n")
    f.write(f"CPU Frequency: {cpu_freq.current:.2f} MHz\n")
    f.write(f"Total RAM: {ram_info.total / (1024 ** 3):.2f} GB\n")
    f.write(f"Available RAM: {ram_info.available / (1024 ** 3):.2f} GB\n")

    # Run nvidia-smi command and capture its output
    try:
        nvidia_smi_output = subprocess.check_output(['nvidia-smi'], universal_newlines=True)
        f.write("\n=== NVIDIA-SMI Output ===\n")
        f.write(nvidia_smi_output)
    except subprocess.CalledProcessError as e:
        f.write("\nError running nvidia-smi: " + str(e))

print(f"Output saved to {output_file}")
