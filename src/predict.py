import joblib
import os
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
MODEL_PATH = os.path.join("models", "myntra_model.pkl")
ENCODER_PATH = os.path.join("models", "color_encoder.pkl")

def predict_user_input():
    print("\n--- 🛍️  Myntra Junior AI Helper  ---")
    
    # 1. Load the Brains
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model not found. Train it first!")
        return

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    
    print("AI Loaded successfully! Ask me about a T-shirt.\n")

    # 2. Get User Input
    while True:
        print("-" * 30)
        try:
            price = float(input("Enter Price (e.g., 500): "))
            color = input("Enter Color (Red, Blue, Black, etc.): ").strip()
            
            # 3. Process Input (Encoding)
            # The model needs the color as a number, not text.
            try:
                # We put it in a list because the encoder expects a list
                color_encoded = encoder.transform([color])[0]
            except ValueError:
                print(f"⚠ Oops! I haven't learned about the color '{color}' yet.")
                print("Try: Red, Blue, Black, White, Pink, Yellow, Green...")
                continue

            # 4. Make Prediction
            # Prepare data: [[price, color_encoded]]
            features = pd.DataFrame([[price, color_encoded]], columns=['price', 'color_encoded'])
            
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1] # Confidence score
            
            # 5. Show Result
            if prediction == 1:
                print(f"\n✅ SUCCESS! The kid will LIKE this t-shirt! (Confidence: {probability:.0%})")
            else:
                print(f"\n❌ PASS. The kid will NOT like this. (Confidence: {1-probability:.0%})")
                
            # Ask to continue
            again = input("\nCheck another item? (y/n): ")
            if again.lower() != 'y':
                break
                
        except ValueError:
            print("Please enter a valid number for price!")

if __name__ == "__main__":
    predict_user_input()