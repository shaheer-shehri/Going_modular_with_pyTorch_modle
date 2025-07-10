"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
from pathlib import Path

import going_modular.data_setup as data_setup
import going_modular.engine as engine
import going_modular.model_builder as model_builder
import going_modular.utils as utils
import torch
from torchvision import transforms

# Setup Hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transform
data_transform = transforms.Compose(
    [transforms.Resize(size=(64, 64)), transforms.ToTensor()]
)

base_dir = Path(__file__).resolve().parent.parent
train_dir = base_dir / "data" / "pizza_steak_sushi" / "train"
test_dir = base_dir / "data" / "pizza_steak_sushi" / "test"
# Create DataLoaders with help of data_setup.py
train_data_loader, test_data_loader, classes_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE,
    num_workers=os.cpu_count(),
)

# Create model using model_builder
model = model_builder.TinyVGG(
    input_shape=3, hidden_units=HIDDEN_UNITS, output_shape=len(classes_names)
).to(device)

# Setup loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# Start Training with help of engine.py
results = engine.train(
    model=model,
    train_dataloader=train_data_loader,
    test_dataloader=test_data_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=NUM_EPOCHS,
    device=device,
)

# Save model using utils.py
utils.save_model(
    model=model,
    target_dir="models",
    model_name="Going_modular_script_mode_tinyVGG_model.pt",
)
