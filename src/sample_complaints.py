import pandas as pd
import os

# Set up paths
BASE_DIR = r"C:\Users\hp\Desktop\Kifiya AIM\week 6\Technical content\workspace"
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_FILE = os.path.join(DATA_DIR, "complaints.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "sampled_complaints.csv")

# Target products
target_products = [
    'Credit card', 'Credit card or prepaid card',
    'Payday loan, title loan, personal loan, or advance loan', 'Buy now, pay later',
    'Checking or savings account', 'Money transfer, virtual currency, or money service'
]

# Load and sample dataset
print("=== Sampling Dataset ===")
try:
    # Read dataset in chunks to manage memory
    chunks = pd.read_csv(INPUT_FILE, low_memory=False, chunksize=100000)
    sampled_dfs = []
    sample_size_per_product = 20000  # Increase to ensure Buy Now Pay Later inclusion
    for chunk in chunks:
        # Filter for target products with non-null narratives
        chunk = chunk[chunk['Product'].isin(target_products) & chunk['Consumer complaint narrative'].notnull()]
        # Sample up to sample_size_per_product per product
        for product in target_products:
            product_chunk = chunk[chunk['Product'] == product]
            if len(product_chunk) > 0:
                sample = product_chunk.sample(n=min(len(product_chunk), sample_size_per_product), random_state=42)
                sampled_dfs.append(sample)
    # Concatenate samples
    df_sampled = pd.concat(sampled_dfs)
    # Cap at 400,000 rows, prioritizing Buy Now Pay Later
    if len(df_sampled) > 400000:
        bnpl_rows = df_sampled[df_sampled['Product'] == 'Buy now, pay later']
        other_rows = df_sampled[df_sampled['Product'] != 'Buy now, pay later']
        if len(bnpl_rows) > 0:
            other_rows = other_rows.sample(n=(400000 - len(bnpl_rows)), random_state=42)
            df_sampled = pd.concat([bnpl_rows, other_rows])
        else:
            df_sampled = df_sampled.sample(n=400000, random_state=42)
    print(f"Sampled dataset shape: {df_sampled.shape}")
    print("Sampled product distribution:")
    print(df_sampled['Product'].value_counts())
    print("Rows with narratives:", len(df_sampled[df_sampled['Consumer complaint narrative'].notnull()]))
except Exception as e:
    print(f"Error sampling dataset: {e}")
    exit()

# Save sampled dataset
try:
    df_sampled.to_csv(OUTPUT_FILE, index=False)
    print(f"Sampled dataset saved to {OUTPUT_FILE}")
except Exception as e:
    print(f"Error saving sampled dataset: {e}")
    exit()