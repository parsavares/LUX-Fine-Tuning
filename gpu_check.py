import torch
import torch.nn as nn

# Define a simple model
model = nn.Linear(100, 10)

# Use DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

# Move model to GPUs
model = model.to('cuda')

# Run a simple forward pass with random input
inputs = torch.randn(16, 100).to('cuda')
outputs = model(inputs)
print(f"Output size: {outputs.size()}")
