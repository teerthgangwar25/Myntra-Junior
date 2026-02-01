import streamlit as st
import pandas as pd
import joblib
import os

# this code has generated with the help of ai as i dont know anything about UI

# --- PAGE SETUP ---
st.set_page_config(page_title="Myntra Junior AI", page_icon="👕")

# --- LOAD BRAIN ---
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "myntra_model.pkl")
    encoder_path = os.path.join("models", "color_encoder.pkl")
    
    if not os.path.exists(model_path):
        return None, None
    
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return model, encoder

model, encoder = load_model()

# --- UI DESIGN ---
st.title("👕 Myntra Junior: AI Designer Helper")
st.markdown("Use this tool to predict if a new **Kids T-Shirt** will be a Hit or a Miss.")

if model is None:
    st.error("❌ Model not found! Please run `src/train_model.py` first.")
else:
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        price = st.number_input("Price (₹)", min_value=100, max_value=5000, value=500, step=50)
        
    with col2:
        # We manually list colors the model knows 
        # (In a real app, we would read this from the encoder)
        known_colors = ['Red', 'Blue', 'Black', 'White', 'Pink', 'Yellow', 'Orange', 'Green', 'Purple', 'Navy Blue']
        color = st.selectbox("Color", known_colors)

    # Prediction Logic
    if st.button("Predict User Reaction", type="primary"):
        # 1. Encode Color
        try:
            color_encoded = encoder.transform([color])[0]
            
            # 2. Predict
            input_data = pd.DataFrame([[price, color_encoded]], columns=['price', 'color_encoded'])
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            
            # 3. Show Result
            st.divider()
            if prediction == 1:
                st.success(f"### ✅ SUCCESS! This item will likely be LOVED.")
                st.write(f"Confidence Score: **{probability:.1%}**")
                st.balloons() # Fun animation!
            else:
                st.error(f"### ❌ PASS. This item might not sell well.")
                st.write(f"Confidence Score: **{1-probability:.1%}** (likelihood of dislike)")
                
        except Exception as e:
            st.error(f"Error: {e}")

# --- SIDEBAR INFO ---
st.sidebar.header("Project Info")
st.sidebar.info(
    """
    **Model:** Logistic Regression
    **Accuracy:** 83%
    **Data:** Synthetic Kids Fashion Data
    **Dev:** You!
    """
)