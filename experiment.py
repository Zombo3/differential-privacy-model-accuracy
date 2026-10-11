
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from opacus import PrivacyEngine

# Reproducibility
torch.manual_seed(9)
start = time.time()

# Experiment settings
N_PUBLIC = 500
N_PRIVATE = 2000
N_TEST = 1000
PUBLIC_EPOCHS = 20
PRIVATE_EPOCHS = 20
TARGET_EPSILON = 8.0
DELTA = 1e-5
BATCH_SIZE = 64

device = torch.device("cpu")
print("Running on:", device, flush=True)

# 1. Load MNIST
transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# 2. Create fixed, non-overlapping datasets
indices = torch.randperm(
    len(train_data),
    generator=torch.Generator().manual_seed(9)
)

# IMPORTANT: Private samples always stay the same
private_indices = indices[:N_PRIVATE].tolist()

# Public samples come AFTER the private samples
public_indices = indices[
    N_PRIVATE:N_PRIVATE + N_PUBLIC
].tolist()

private_data = Subset(train_data, private_indices)
public_data = Subset(train_data, public_indices)

# Fixed test dataset
test_subset = Subset(
    test_data,
    list(range(N_TEST))
)

print(f"Private samples: {len(private_data)}")
print(f"Public samples: {len(public_data)}")
print(f"Test samples: {len(test_subset)}")

# 3. Data loaders
public_loader = DataLoader(
    public_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

private_loader = DataLoader(
    private_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_subset,
    batch_size=256,
    shuffle=False
)

# 4. Small neural network
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
).to(device)

loss_fn = nn.CrossEntropyLoss()

# 5. Public pretraining (non-private SGD)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)

print("Starting public pretraining...", flush=True)

for epoch in range(PUBLIC_EPOCHS):
    model.train()

    for images, labels in public_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()

print("Public pretraining complete!", flush=True)

# 6. Private fine-tuning with DP-SGD
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)

privacy_engine = PrivacyEngine()

model, optimizer, private_loader = (
    privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=private_loader,
        epochs=PRIVATE_EPOCHS,
        target_epsilon=TARGET_EPSILON,
        target_delta=DELTA,
        max_grad_norm=1.0
    )
)

print("Starting DP-SGD training...", flush=True)

for epoch in range(PRIVATE_EPOCHS):
    model.train()

    for images, labels in private_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()

# Calculate actual privacy expenditure
epsilon = privacy_engine.get_epsilon(DELTA)

# 7. Evaluate on unchanged test data
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images).argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = 100.0 * correct / total
elapsed = time.time() - start

# 8. Display results
print("\n===== EXPERIMENT RESULTS =====")
print(f"Public samples: {N_PUBLIC}")
print(f"Private samples: {N_PRIVATE}")
print(f"Public epochs: {PUBLIC_EPOCHS}")
print(f"Private epochs: {PRIVATE_EPOCHS}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Actual epsilon: {epsilon:.4f}")
print(f"Delta: {DELTA}")
print(f"Training time: {elapsed:.1f} seconds")
print("==============================")
