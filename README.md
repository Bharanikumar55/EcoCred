# 🌱 EcoCred – Eco-Friendly Loan Eligibility and Default Prediction System

> 🧪 **Status**: _Under Development_  
> 📦 **Version**: v0.1  
> 🔗 Live Demo: _Coming Soon_

---

## 📝 Abstract

**EcoCred** is an intelligent, eco-conscious loan approval system designed to assess a user’s loan eligibility based not only on financial credentials (like income, loan amount, credit score) but also their eco-friendly behavior — such as electricity consumption and vehicle fuel type. By integrating OCR, ML, and a web-based interface, it aims to incentivize sustainability while solving real-world financial challenges.

---

## 🎯 Objectives

- ✅ Predict loan default risk using ML models.
- ✅ Calculate an **eco score** based on fuel type and power usage.
- ✅ Use OCR to extract key values from uploaded electricity bills and RCs.
- ✅ Provide users with personalized loan approval and recommendations.
- ✅ Build an interactive web interface for real-world simulation.

---

## 🔄 Methodology

### 📊 Data Sources
- Lending Club dataset (cleaned)
- Simulated eco-friendly values (e.g., electricity units)
- Manually created sample RC and electricity bill images

### ⚙️ ML Models
- Logistic Regression
- Random Forest
- XGBoost (experimental)

### 🧠 Input Features
- Income
- Loan Amount
- Electricity Units
- Vehicle Type
- `eco_score` (calculated from uploaded documents)

### 🔍 OCR + Image Preprocessing
- Pytesseract used to extract:
  - ⚡ kWh units from electricity bills
  - ⛽ Fuel type (Petrol / Diesel / Electric) from RC
- Preprocessing includes: grayscale, thresholding, noise removal

---

## 🧩 Project Modules

| Module                | Description                                                  |
|-----------------------|--------------------------------------------------------------|
| 👤 User Input Module  | Collects income, loan amount, electricity bill & RC images   |
| 🧾 OCR Module         | Extracts values from uploaded images                         |
| 🌍 Eco Scoring Module | Calculates eco score based on fuel type & electricity usage |
| 🤖 ML Prediction      | Predicts loan approval (0 = reject, 1 = approve)             |
| 📊 Result Display     | Shows outcome, eco score, and risk analysis                  |

---

## 🧰 Tech Stack

| Category       | Tools/Libraries                         |
|----------------|------------------------------------------|
| Language       | Python                                   |
| ML Libraries   | `sklearn`, `pandas`, `numpy`, `matplotlib` |
| OCR            | `pytesseract`, `opencv`                 |
| Web Framework  | `Flask` (backend), `HTML/CSS/JS` (frontend) |
| DB (planned)   | `MongoDB` (for tracking user loan history) |
| Version Control| Git, GitHub                              |

---

## 💡 Features (Implemented & Upcoming)

✅ Trained ML models for loan prediction  
✅ Eco score integration based on document OCR  
✅ Upload interface for electricity bill & RC  
✅ OCR using `pytesseract`  
🕓 MongoDB integration for loan history (in progress)  
🕓 Google login and user-based personalization (planned)  
🕓 Credit history-based score boosting (planned)  
🕓 PDF download of loan approval (planned)

---

## 📈 Sample Inputs

| Feature            | Example Values                      |
|--------------------|-------------------------------------|
| Monthly Income     | ₹65,000                             |
| Loan Amount        | ₹25,000                             |
| Electricity Units  | 220 kWh (from bill via OCR)         |
| Vehicle Type       | Petrol / Diesel / Electric (from RC)|
| Eco Score          | 0 to 100 (auto-calculated)          |

---

## 🚀 Future Enhancements

- 🔗 Google login + dashboard to auto-fill returning user data
- 🧾 Loan status tracker with MongoDB backend
- 📲 Aadhaar-based verification
- 🤖 Deep learning-based OCR for better extraction
- 📤 Loan approval letter (PDF) with auto-generated loan ID
- 📬 EMI tracking & reminder system

---

## 📁 Folder Structure

```bash
EcoCred/
│
├── app.py                  # Flask backend logic
├── templates/              # HTML frontend
├── static/                 # CSS, images
├── notebooks/              # All Jupyter notebooks (ML & OCR)
├── .gitignore              # File exclusion rules
└── README.md               # This file
