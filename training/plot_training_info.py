import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("../runs/sae/layer8_bt8/metrics.csv")

# Separate splits
train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "val"]

metrics = [
    ("recon_mse", "Reconstruction MSE"),
    ("sparsity_l1", "L1 Sparsity Loss"),
    ("total_loss", "Total Loss"),
    ("mean_active", "Mean Active Units"),
]

for col, title in metrics:
    plt.figure()

    if not train_df.empty:
        plt.plot(train_df["step"], train_df[col], label="train")

    if not val_df.empty:
        plt.plot(val_df["step"], val_df[col], label="validation")

    plt.xlabel("Step")
    plt.ylabel(title)
    plt.title(f"{title} vs Step")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"../runs/sae/layer8_bt8/plots/{col}.png")