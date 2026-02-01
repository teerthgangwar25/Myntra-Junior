# 👕 Myntra Junior: AI Fashion Predictor

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/ML-Logistic%20Regression-orange)
![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-9cf)

**Myntra Junior** is an End-to-End Machine Learning project that predicts whether a kid (or parent) will like a specific t-shirt based on its **Price** and **Color**.

It features a complete MLOps pipeline, including synthetic data generation, model training, version control with DVC, and an interactive web UI.

---

## The Project Story: From Idea to MLOps

### 1. The Spark 💡
I wanted to build something real, moving beyond standard datasets like "Titanic." My idea was **Myntra Junior**: A smart assistant to help parents find affordable, stylish clothes for kids.
* **Hypothesis:** Can we predict "Likability" based purely on Price and Color?

### 2. The Roadblock 
I started by downloading a massive real-world Fashion Dataset.
* **The Failure:** The data was dominated by adult clothing (Men/Women). When filtering for "Kids T-shirts," I was left with almost zero data points.
* **The Lesson:** Real-world data is often messy or missing.

### 3. The Pivot: Data Engineering 
Instead of quitting, I wrote a **Synthetic Data Generator**. I created 5,000 rows of data based on realistic business logic:
* **Rule A:** Parents prefer items under ₹800.
* **Rule B:** Kids prefer bright colors (Red, Yellow) over dull ones (Grey, Black).

### 4. The Solution 
* **Model:** I trained a **Logistic Regression** classifier on this data.
* **Accuracy:** The model achieved **~83% accuracy**, successfully learning the hidden rules I programmed.
* **UI:** I built a **Streamlit App** so users can interact with the model without touching code.

---

## Project Structure

I organized this project using industry-standard MLOps practices, separating code, data, and models. The structure ensures that heavy files are tracked via DVC while code is versioned in Git.

```text
MYNTRA-JUNIOR-PROJECT/
│
├── .dvc/                  # DVC internal configuration
├── .dvcignore             # Files for DVC to ignore
├── .gitignore             # Files for Git to ignore
│
├── data/                  # Data storage (Tracked by DVC)
│   ├── processed/         
│   │   ├── myntra_junior_clean.csv      # Cleaned real data
│   │   └── myntra_junior_synthetic.csv  # Generated synthetic data
│   └── raw/
│       └── Fashion Dataset.csv/         # Original dataset folder
│           └── Myntra fashion product dataset.csv
│
├── models/                # Model artifacts (Tracked by DVC)
│   ├── color_encoder.pkl  # Encoder to translate colors -> numbers
│   ├── metrics.txt        # Performance report (Accuracy score)
│   └── myntra_model.pkl   # The trained Logistic Regression model
│
├── src/                   # Source Code
│   ├── app.py             # Streamlit Frontend Web App
│   ├── data_cleaning.py   # Script to clean & filter raw data
│   ├── generate_data.py   # Script to create synthetic data
│   ├── predict.py         # Terminal-based prediction logic
│   └── train_model.py     # Script to train & save the model
│
├── data.dvc               # DVC pointer file for the 'data' folder
├── models.dvc             # DVC pointer file for the 'models' folder
└── requirements.txt       # List of Python dependencies


**Disclaimer:** This is a personal project created for educational purposes. It is not affiliated with, endorsed by, or connected to Myntra or any of its subsidiaries. The dataset used is synthetic/publicly available.
