import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURATION ---
PROCESSED_DATA_PATH = os.path.join("data", "processed", "myntra_junior_synthetic.csv")
MODEL_PATH = os.path.join("models", "myntra_model.pkl")
METRICS_PATH = os.path.join("models", "metrics.txt")
ENCODER_PATH = os.path.join("models", "color_encoder.pkl")

def train():
    print("Loading clean data...")
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError("❌ Error: Clean data not found. Run data_cleaning.py first!")

    df = pd.read_csv(PROCESSED_DATA_PATH)

    # 1. Prepare Data (Feature Engineering)
    # The model can't read text like "Navy Blue". We convert it to numbers.
    print("Encoding text columns...")
    encoder = LabelEncoder()
    df['color_encoded'] = encoder.fit_transform(df['color'])
    
    # Save the encoder (We need this to make predictions later!)
    os.makedirs("models", exist_ok=True)
    joblib.dump(encoder, ENCODER_PATH)

    # We use Price and Color to predict 'Liked'
    # We drop columns like 'product_name' because they are just text labels, not features to learn from
    X = df[['price', 'color_encoded']] 
    y = df['liked']

    # 2. Split Data (80% for training, 20% for testing)
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train the Model
    print("Training Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 4. Evaluate
    print("Evaluating...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✔ Model Accuracy: {accuracy:.2f}")

    # 5. Save the Brain
    joblib.dump(model, MODEL_PATH)
    print(f"✔ Model saved to {MODEL_PATH}")

    # Save a report card
    with open(METRICS_PATH, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred))

if __name__ == "__main__":
    try:
        train()
        print("\nSUCCESS! Phase 2 Complete.")
    except Exception as e:
        print(f"❌ Error: {e}")