# Placement Prediction System (Machine Learning)

A Machine Learning-based classification web application that predicts student campus placement outcomes using **Logistic Regression**.

---

## 📌 Features
- **Prediction Engine**: Logistic Regression trained on academic metrics (CGPA & IQ).
- **Interactive UI**: Clean, lightweight, and fast dashboard built with **Streamlit**.
- **Real-Time Probability**: Shows likelihood of placement with confidence percentage.

---

## 📂 Project Structure
```
├── app.py                      # Streamlit Web Application
├── model.pkl                   # Trained Logistic Regression Model
├── end-to-end-toy-project.ipynb # Training & Preprocessing Notebook
├── tensor.ipynb                # Supplementary Notebook
├── requirements.txt            # Python Dependencies
├── .gitignore                  # Ignored files (including datasets)
└── README.md                   # Project Documentation
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.
