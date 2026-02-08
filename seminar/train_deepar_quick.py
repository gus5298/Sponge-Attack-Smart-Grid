"""Quick DeepAR training with minimal epochs for testing."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (CONTEXT_LEN, PREDICTION_LEN, NUM_FEATURES, 
                    DEEPAR_MODEL_PATH, ALL_FEATURES)
from models.deepar import DeepARLSTM
from utils.dataset import TimeSeriesDataset
from utils.data_loader import load_all_locations, get_normalization_params

# Reduced parameters for fast training
QUICK_HIDDEN_SIZE = 64
QUICK_RNN_LAYERS = 1
QUICK_EPOCHS = 3
QUICK_BATCH_SIZE = 128

print("=" * 50)
print("QUICK DeepAR Training (3 epochs)")
print("=" * 50)

print("\nLoading data...")
combined_df = load_all_locations("data")
data = combined_df[ALL_FEATURES].values.astype(np.float32)
mean, std = get_normalization_params(data)
data = (data - mean) / std
norm_params = {'mean': mean.tolist(), 'std': std.tolist()}

# Use smaller subset for quick training
train_data = data[:5000]
val_data = data[5000:6000]

train_dataset = TimeSeriesDataset(train_data, CONTEXT_LEN, PREDICTION_LEN)
val_dataset = TimeSeriesDataset(val_data, CONTEXT_LEN, PREDICTION_LEN)
train_loader = DataLoader(train_dataset, batch_size=QUICK_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=QUICK_BATCH_SIZE, shuffle=False)

print(f"Training samples: {len(train_dataset)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = DeepARLSTM(
    input_size=NUM_FEATURES,
    hidden_size=QUICK_HIDDEN_SIZE,
    num_layers=QUICK_RNN_LAYERS,
    output_size=PREDICTION_LEN,
    dropout=0.0
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(QUICK_EPOCHS):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    print(f"Epoch {epoch+1}/{QUICK_EPOCHS}: Loss={train_loss:.6f}")

# Save with expected structure
torch.save({
    'model_state_dict': model.state_dict(),
    'context_len': CONTEXT_LEN,
    'prediction_len': PREDICTION_LEN,
    'hidden_size': QUICK_HIDDEN_SIZE,
    'rnn_layers': QUICK_RNN_LAYERS,
    'num_features': NUM_FEATURES,
    'feature_cols': ALL_FEATURES,
    'norm_params': norm_params,
}, DEEPAR_MODEL_PATH)

print(f"\nModel saved to {DEEPAR_MODEL_PATH}")
print("=" * 50)
