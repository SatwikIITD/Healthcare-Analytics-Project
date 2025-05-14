import pandas as pd
import numpy as np
import os

os.makedirs("nih_rm",exist_ok=True)
os.makedirs("rm_indices",exist_ok=True)

np.random.seed(42)

for i in range(2013,2022):
    # Load dataset
    df = pd.read_csv(f"nih/sparcs_{i}.csv")  # Replace with your actual file

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Compute number of rows to remove
    num_rows_to_remove = int(0.1 * len(df))

    # Randomly select row indices to remove
    removed_indices = np.random.choice(df.index, num_rows_to_remove, replace=False)

    # Save removed row indices in a compressed format
    np.savez_compressed(f"rm_indices/sparcs_{i}.npz", removed_indices=removed_indices)

    # Remove selected rows
    df_cleaned = df.drop(removed_indices).reset_index(drop=True)

    # Save the cleaned dataset
    df_cleaned.to_csv(f"nih_rm/sparcs_{i}.csv", index=False)


