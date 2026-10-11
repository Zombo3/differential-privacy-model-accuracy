
import csv
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from opacus import PrivacyEngine

# ==============================
# EXPERIMENT SETTINGS
# ==============================

PUBLIC_EPOCHS = 20
PRIVATE_EPOCHS = 20

N_PRIVATE = 2000
N_TEST = 1000

TARGET_EPSILON = 8.0
DELTA = 1e-5
BATCH_SIZE = 64

SEEDS = [9, 21, 42]
PUBLIC_SIZES = [500, 1000, 2000, 5000, 9000]
ROTATIONS = [0, 15, 30, 45, 60]

DEVICE = torch.device("cpu")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

print("Running on:", DEVICE, flush=True)

# ==============================
# LOAD MNIST ONCE
# ==============================

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# Fix the dataset split for every run.
split_generator = torch.Generator().manual_seed(9)
indices = torch.randperm(
    len(train_data), generator=split_generator
).tolist()

private_indices = indices[:N_PRIVATE]
public_pool = indices[
    N_PRIVATE:N_PRIVATE + max(PUBLIC_SIZES)
]
test_indices = list(range(N_TEST))

assert not set(private_indices).intersection(public_pool)

# ==============================
# DATASET WRAPPER
# ==============================

class ImageSubset(Dataset):
    def __init__(self, dataset, indices, rotation=0):
        self.dataset = dataset
        self.indices = indices
        self.rotation = rotation

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]

        if self.rotation != 0:
            image = TF.rotate(
                image,
                angle=self.rotation,
                interpolation=InterpolationMode.BILINEAR,
                fill=0
            )

        return image, label


private_data = ImageSubset(train_data, private_indices)
test_subset = ImageSubset(test_data, test_indices)

test_loader = DataLoader(
    test_subset,
    batch_size=256,
    shuffle=False
)

# ==============================
# MODEL
# ==============================

def create_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    ).to(DEVICE)


# ==============================
# SINGLE EXPERIMENT
# ==============================

def run_experiment(n_public, rotation, seed):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    start_time = time.time()

    public_data = ImageSubset(
        train_data,
        public_pool[:n_public],
        rotation=rotation
    )

    public_loader = DataLoader(
        public_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1000)
    )

    private_loader = DataLoader(
        private_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = create_model()
    loss_fn = nn.CrossEntropyLoss()

    # PUBLIC PRETRAINING
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1
    )

    for epoch in range(PUBLIC_EPOCHS):
        model.train()

        for images, labels in public_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()

    # PRIVATE DP-SGD TRAINING
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1
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

    for epoch in range(PRIVATE_EPOCHS):
        model.train()

        for images, labels in private_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()

    epsilon = privacy_engine.get_epsilon(DELTA)

    # EVALUATION
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    elapsed = time.time() - start_time

    return {
        "public_samples": n_public,
        "rotation": rotation,
        "seed": seed,
        "accuracy": accuracy,
        "epsilon": epsilon,
        "delta": DELTA,
        "time_seconds": elapsed
    }


# ==============================
# BUILD 27 RUNS
# ==============================

configurations = []

# Experiment A: amount of public data
for size in PUBLIC_SIZES:
    configurations.append((size, 0))

# Experiment B: rotation.
# The 5000-sample / 0-degree reference is already included.
for angle in ROTATIONS:
    if angle != 0:
        configurations.append((5000, angle))

runs = [
    (size, angle, seed)
    for size, angle in configurations
    for seed in SEEDS
]

assert len(runs) == 27

fields = [
    "public_samples",
    "rotation",
    "seed",
    "accuracy",
    "epsilon",
    "delta",
    "time_seconds"
]

raw_path = RESULTS_DIR / "all_runs.csv"

# Save results after each successful run
with open(raw_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    for i, (size, angle, seed) in enumerate(runs, start=1):
        print(
            f"\nRun {i}/27 | Public={size} "
            f"| Rotation={angle} | Seed={seed}",
            flush=True
        )

        result = run_experiment(size, angle, seed)
        writer.writerow(result)
        file.flush()

        print(
            f"Accuracy: {result['accuracy']:.2f}% | "
            f"Epsilon: {result['epsilon']:.4f} | "
            f"Time: {result['time_seconds']:.1f}s",
            flush=True
        )

# ==============================
# SUMMARIZE RESULTS
# ==============================

import pandas as pd

df = pd.read_csv(raw_path)

summary = (
    df.groupby(["public_samples", "rotation"])
      .agg(
          mean_accuracy=("accuracy", "mean"),
          std_accuracy=("accuracy", "std"),
          mean_epsilon=("epsilon", "mean"),
          mean_time=("time_seconds", "mean"),
          repetitions=("seed", "count")
      )
      .reset_index()
)

summary.to_csv(
    RESULTS_DIR / "summary.csv",
    index=False
)

# ==============================
# GRAPH 1: PUBLIC DATA AMOUNT
# ==============================

amount = summary[summary["rotation"] == 0].sort_values(
    "public_samples"
)

plt.figure(figsize=(8, 5))
plt.errorbar(
    amount["public_samples"],
    amount["mean_accuracy"],
    yerr=amount["std_accuracy"],
    marker="o",
    capsize=4
)
plt.xlabel("Number of public MNIST samples")
plt.ylabel("Student test accuracy (%)")
plt.title("DP-SGD Accuracy vs. Public Data Amount")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "accuracy_vs_public_amount.png",
    dpi=200
)
plt.close()

# ==============================
# GRAPH 2: DISTRIBUTION SHIFT
# ==============================

rotation = summary[
    summary["public_samples"] == 5000
].sort_values("rotation")

plt.figure(figsize=(8, 5))
plt.errorbar(
    rotation["rotation"],
    rotation["mean_accuracy"],
    yerr=rotation["std_accuracy"],
    marker="o",
    capsize=4
)
plt.xlabel("Public image rotation (degrees)")
plt.ylabel("Student test accuracy (%)")
plt.title("DP-SGD Accuracy vs. Public Data Rotation")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "accuracy_vs_rotation.png",
    dpi=200
)
plt.close()

print("\n===== ALL EXPERIMENTS COMPLETE =====")
print(summary.to_string(index=False))
print("\nFiles saved in the results/ folder:")
print(" - all_runs.csv")
print(" - summary.csv")
print(" - accuracy_vs_public_amount.png")
print(" - accuracy_vs_rotation.png")
