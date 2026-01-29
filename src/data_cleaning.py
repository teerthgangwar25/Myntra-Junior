import pandas as pd
import os

# --- CONFIGURATION ---
RAW_DATA_PATH = os.path.join("data", "raw", "Fashion Dataset.csv", "Myntra fashion product dataset.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "myntra_junior_clean.csv")

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Error: Could not find file at {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"✔ Data loaded. Shape: {df.shape}")
    return df

def clean_data(df):
    print("Starting data cleaning...")
    
    # 1. Standardize columns
    df.columns = df.columns.str.strip()
    
    # Rename columns to what our model expects
    # (We map 'avg_rating' to 'rating' and 'colour' to 'color')
    # Note: We use .get() so it doesn't crash if a column is missing
    df = df.rename(columns={
        'name': 'product_name', 
        'price': 'price', 
        'colour': 'color', 
        'avg_rating': 'rating'
    })

    # 2. Filter for Kids (Safe Mode)
    kids_mask = df['product_name'].str.contains('Boy|Girl|Kid|Junior', case=False, na=False)
    kids_df = df[kids_mask]
    
    if len(kids_df) > 100:
        print(f"   -> Found {len(kids_df)} Kids' items. Keeping them.")
        df = kids_df
    else:
        print(f"   ⚠ Warning: Only found {len(kids_df)} Kids' items. Using ALL data instead to prevent crash.")

    # 3. Filter for T-Shirts (Safe Mode)
    tshirt_mask = df['product_name'].str.contains('T-shirt', case=False, na=False)
    tshirt_df = df[tshirt_mask]

    if len(tshirt_df) > 50:
        print(f"   -> Found {len(tshirt_df)} T-shirts. Keeping them.")
        df = tshirt_df
    else:
        print(f"   ⚠ Warning: Only found {len(tshirt_df)} T-shirts. Using other items too.")

    # 4. Final Column Selection
    # Ensure we only keep rows that actually have prices and ratings
    df = df[['price', 'color', 'rating']].dropna()

    # 5. CREATE TARGET VARIABLE: 'liked'
    df['liked'] = df['rating'].apply(lambda x: 1 if x > 4.1 else 0)
    
    print(f"✔ Cleaning complete. Final shape: {df.shape}")
    return df

def save_data(df, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✔ File saved successfully at: {filepath}")

if __name__ == "__main__":
    try:
        raw_df = load_data(RAW_DATA_PATH)
        clean_df = clean_data(raw_df)
        if len(clean_df) == 0:
            print("❌ Error: The dataset is empty after cleaning! Check your CSV file content.")
        else:
            save_data(clean_df, PROCESSED_DATA_PATH)
            print("\nSUCCESS! You are ready for Phase 2.")
    except Exception as e:
        print(f"❌ Something went wrong: {e}")