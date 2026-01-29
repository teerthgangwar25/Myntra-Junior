import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
OUTPUT_PATH = os.path.join("data", "processed", "myntra_junior_synthetic.csv")
NUM_SAMPLES = 5000  # We will generate 5,000 rows (Plenty for a good score!)

def generate_synthetic_data():
    print(f"Generating {NUM_SAMPLES} rows of synthetic data for Myntra Junior...")
    
    np.random.seed(42) # Ensures we get the same random numbers every time

    # 1. Generate Random Features
    # IDs
    ids = range(1, NUM_SAMPLES + 1)
    
    # Colors: Kids like bright colors! 
    # We'll make bright colors more common to simulate a kids store inventory
    colors = ['Black', 'White', 'Navy Blue', 'Red', 'Pink', 'Yellow', 'Orange', 'Green', 'Purple', 'Blue']
    color_data = np.random.choice(colors, NUM_SAMPLES)
    
    # Prices: Between 199 and 1500
    price_data = np.random.randint(199, 1500, NUM_SAMPLES)
    
    # Brands (Just for flavor)
    brands = ['H&M Kids', 'Max', 'Pantaloons Jr', 'Gap Kids', 'Mothercare', 'Zara Kids']
    brand_data = np.random.choice(brands, NUM_SAMPLES)

    # 2. Define the "Hidden Rules" (The Logic)
    # This is where we simulate "User Behavior". 
    # Let's say: Kids/Parents LIKE items that are Cheap OR Brightly Colored.
    
    liked_data = []
    
    for i in range(NUM_SAMPLES):
        score = 0
        current_color = color_data[i]
        current_price = price_data[i]
        
        # RULE 1: Price Preference (Parents love cheap clothes)
        if current_price < 600:
            score += 3
        elif current_price < 900:
            score += 1
        else:
            score -= 2 # Too expensive!
            
        # RULE 2: Color Preference (Kids love bright colors)
        if current_color in ['Red', 'Pink', 'Yellow', 'Orange', 'Purple']:
            score += 3
        elif current_color in ['Black', 'Navy Blue']:
            score -= 1 # Boring for some kids
            
        # Add some randomness (Human behavior is unpredictable)
        noise = np.random.normal(0, 1)
        final_score = score + noise
        
        # Determine Outcome
        if final_score > 1.5:
            liked_data.append(1) # Liked
        else:
            liked_data.append(0) # Not Liked

    # 3. Save to DataFrame
    df = pd.DataFrame({
        'product_id': ids,
        'brand': brand_data,
        'price': price_data,
        'color': color_data,
        'liked': liked_data
    })
    
    # Show stats
    print("\nData Distribution:")
    print(df['liked'].value_counts())
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✔ Success! Synthetic data saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_synthetic_data()