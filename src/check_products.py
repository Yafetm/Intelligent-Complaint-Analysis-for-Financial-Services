import pandas as pd
import os
DATA_DIR = r"C:\Users\hp\Desktop\Kifiya AIM\week 6\Technical content\workspace\data"
INPUT_FILE = os.path.join(DATA_DIR, "complaints.csv")
df = pd.read_csv(INPUT_FILE, low_memory=False)
print("Unique Product Names:")
print(df['Product'].unique())