# 👕 Myntra Junior: AI Fashion Predictor

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/ML-Logistic%20Regression-orange)
![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-9cf)

**Myntra Junior** is an End-to-End Machine Learning project that predicts whether a kid (or parent) will like a specific t-shirt based on its **Price** and **Color**.

It features a complete MLOps pipeline, including synthetic data generation, model training, version control with DVC, and an interactive web UI.

---

## 📖 The Project Story: From Idea to MLOps

### 1. The Spark 💡
I wanted to build something real, moving beyond standard datasets like "Titanic." My idea was **Myntra Junior**: A smart assistant to help parents find affordable, stylish clothes for kids.
* **Hypothesis:** Can we predict "Likability" based purely on Price and Color?

### 2. The Roadblock 🚧
I started by downloading a massive real-world Fashion Dataset.
* **The Failure:** The data was dominated by adult clothing (Men/Women). When filtering for "Kids T-shirts," I was left with almost zero data points.
* **The Lesson:** Real-world data is often messy or missing.

### 3. The Pivot: Data Engineering 🛠️
Instead of quitting, I wrote a **Synthetic Data Generator**. I created 5,000 rows of data based on realistic business logic:
* **Rule A:** Parents prefer items under ₹800.
* **Rule B:** Kids prefer bright colors (Red, Yellow) over dull ones (Grey, Black).

### 4. The Solution 🧠
* **Model:** I trained a **Logistic Regression** classifier on this data.
* **Accuracy:** The model achieved **~83% accuracy**, successfully learning the hidden rules I programmed.
* **UI:** I built a **Streamlit App** so users can interact with the model without touching code.

---

## 🏗️ Project Structure

I organized this project using industry-standard MLOps practices, separating code, data, and models.

```text
Myntra-Junior/
│
├── .dvc/                  # DVC configuration files
├── .gitignore             # Git ignore file
├── dvc.yaml               # DVC pipeline
├── README.md              # Project Documentation
├── requirements.txt       # Python dependencies
│
├── data/                  # Data storage (Tracked by DVC)
│   ├── raw/               # Original datasets
│   └── processed/         # Cleaned & Synthetic data
│
├── models/                # Binary artifacts (Tracked by DVC)
│   ├── myntra_model.pkl   # The trained AI brain
│   └── color_encoder.pkl  # Translator for text->numbers
│
└── src/                   # Source Code
    ├── app.py             # Streamlit Web App
    ├── data_cleaning.py   # Script to clean raw data
    ├── generate_data.py   # Script to create synthetic data
    ├── train_model.py     # Script to train and save model
    └── predict.py         # Terminal-based prediction tool
