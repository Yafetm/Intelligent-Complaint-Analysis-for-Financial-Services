import pandas as pd
import os
import re
import matplotlib.pyplot as plt

# Set up paths
BASE_DIR = r"C:\Users\hp\Desktop\Kifiya AIM\week 6\Technical content\workspace"
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_FILE = os.path.join(DATA_DIR, "sampled_complaints.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "filtered_complaints.csv")

# Product mapping
product_mapping = {
    'Credit card': 'Credit Card',
    'Credit card or prepaid card': 'Credit Card',
    'Consumer Loan': 'Personal Loan',
    'Payday loan, title loan, or personal loan': 'Personal Loan',
    'Payday loan, title loan, personal loan, or advance loan': 'Personal Loan',
    'Payday loan': 'Personal Loan',
    'Buy now, pay later': 'Buy Now Pay Later',
    'Checking or savings account': 'Savings Account',
    'Money transfer, virtual currency, or money service': 'Money Transfers',
    'Money transfers': 'Money Transfers'
}

# Step 1: Load dataset
print("=== Loading Dataset ===")
try:
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Step 2: EDA
print("\n=== Exploratory Data Analysis ===")
print("Product distribution (before mapping):")
print(df['Product'].value_counts())
df['Product'] = df['Product'].map(product_mapping).fillna(df['Product'])
print("\nProduct distribution (after mapping):")
print(df['Product'].value_counts())
df['narrative_length'] = df['Consumer complaint narrative'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)
print("\nNarrative length statistics:")
print(df['narrative_length'].describe())
plt.hist(df['narrative_length'], bins=50, range=(0, 1000))
plt.title('Narrative Length Distribution')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.savefig(os.path.join(BASE_DIR, 'narrative_length_distribution.png'))
plt.close()

# Step 3: Filter dataset
print("\n=== Filtering Dataset ===")
target_products = ['Credit Card', 'Personal Loan', 'Buy Now Pay Later', 'Savings Account', 'Money Transfers']
df_filtered = df[df['Product'].isin(target_products) & df['Consumer complaint narrative'].notnull()].copy()
print(f"Filtered dataset shape: {df_filtered.shape}")
print("Filtered product distribution:")
print(df_filtered['Product'].value_counts())

# Step 4: Clean narratives
print("\n=== Cleaning Narratives ===")
def clean_narrative(text):
    text = str(text).lower()
    text = re.sub(r'xxxx+\b', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    boilerplate = ['i am writing to file a complaint', 'please help', 'consumer complaint']
    for phrase in boilerplate:
        text = text.replace(phrase, '')
    return text.strip()
df_filtered['cleaned_narrative'] = df['Consumer complaint narrative'].apply(clean_narrative)
print("Sample cleaned narratives:")
print(df_filtered['cleaned_narrative'].head(3))

# Step 5: Save filtered dataset
df_filtered.to_csv(OUTPUT_FILE, index=False)
print(f"\nFiltered dataset saved to {OUTPUT_FILE}")