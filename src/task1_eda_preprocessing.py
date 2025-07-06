import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# Set up paths
BASE_DIR = r"C:\Users\hp\Desktop\Kifiya AIM\week 6\Technical content\workspace"
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_FILE = os.path.join(DATA_DIR, "complaints.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "filtered_complaints.csv")

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Step 1: Load the dataset
print("=== Loading Dataset ===")
try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Step 2: EDA
print("\n=== Exploratory Data Analysis ===")
# Distribution of complaints by Product
product_distribution = df['Product'].value_counts()
print("\nComplaint Distribution by Product:")
print(product_distribution.to_string())

# Narrative length analysis
df['narrative_length'] = df['Consumer complaint narrative'].apply(
    lambda x: len(str(x).split()) if pd.notnull(x) else 0
)
print("\nNarrative Length Statistics:")
print(df['narrative_length'].describe().to_string())

# Complaints with and without narratives
narratives_present = df['Consumer complaint narrative'].notnull().sum()
narratives_missing = df['Consumer complaint narrative'].isnull().sum()
print(f"\nComplaints with narratives: {narratives_present}")
print(f"Complaints without narratives: {narratives_missing}")

# Visualize narrative length distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['narrative_length'], bins=50)
plt.title("Distribution of Narrative Lengths")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.savefig(os.path.join(BASE_DIR, "narrative_length_distribution.png"))
plt.close()
print("\nNarrative length distribution plot saved to narrative_length_distribution.png")

# Step 3: Filter dataset
print("\n=== Filtering Dataset ===")
# Map product names to match the specified categories (adjust based on actual dataset)
product_mapping = {
    'Credit card': 'Credit Card',
    'Credit card or prepaid card': 'Credit Card',
    'Consumer Loan': 'Personal Loan',
    'Payday loan, title loan, or personal loan': 'Personal Loan',
    'Buy now, pay later': 'Buy Now Pay Later',
    'Checking or savings account': 'Savings Account',
    'Money transfer, virtual currency, or money service': 'Money Transfers'
}
df['Product'] = df['Product'].replace(product_mapping)

# Filter for specified products
target_products = ['Credit Card', 'Personal Loan', 'Buy Now Pay Later', 'Savings Account', 'Money Transfers']
df_filtered = df[df['Product'].isin(target_products)]

# Remove records with empty narratives
df_filtered = df_filtered[df_filtered['Consumer complaint narrative'].notnull()]
print(f"Filtered dataset shape: {df_filtered.shape}")
print(f"Filtered products: {df_filtered['Product'].unique()}")

# Step 4: Clean narratives
print("\n=== Cleaning Narratives ===")
def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove special characters, keep alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Remove boilerplate (example phrases, adjust as needed)
    boilerplate = [
        'i am writing to file a complaint',
        'please assist',
        'thank you for your attention'
    ]
    for phrase in boilerplate:
        text = text.replace(phrase, '')
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

df_filtered['cleaned_narrative'] = df_filtered['Consumer complaint narrative'].apply(clean_text)
print("Sample cleaned narratives:")
for i, narrative in enumerate(df_filtered['cleaned_narrative'].head(5), 1):
    print(f"Narrative {i}: {narrative[:100]}...")  # Truncate for brevity

# Step 5: Save filtered dataset
df_filtered.to_csv(OUTPUT_FILE, index=False)
print(f"\nFiltered dataset saved to {OUTPUT_FILE}")
print(f"Final filtered dataset shape: {df_filtered.shape}")