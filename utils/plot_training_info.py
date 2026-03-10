import pandas as pd
import matplotlib.pyplot as plt
import os

log_dir = "../trained_models/layer8_8192_batchtopk_32_0.0003"

os.makedirs(f"{log_dir}/plots", exist_ok=True)
# Load CSV
df = pd.read_csv(f"{log_dir}/logs.csv")
# Separate splits
# train_df = df[df["split"] == "train"]
# val_df = df[df["split"] == "val"]

# train_df = train_df[-200:]
# val_df = val_df[val_df["step"].isin(train_df["step"])]

metrics = [
    ("total_loss", "Total Loss"),
    ("l2_loss", "Reconstruction MSE"),
    ("l1_loss", "L1 Sparsity Loss"),
    ("l1_norm", "L1 Norm"),
    ("aux_loss", "Auxiliary Loss"),
    ("num_dead_features", "Number of Dead Features"),
]

for col, title in metrics:
    plt.figure()

    if not df.empty:
        plt.plot(df["step"], df[col])

    
    plt.xlabel("Step")
    plt.ylabel(title)
    plt.title(f"{title} vs Step")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    plt.savefig(f"../trained_models/layer8_8192_batchtopk_32_0.0003/plots/{col}.png")