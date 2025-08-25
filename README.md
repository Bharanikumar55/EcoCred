# 🌱 EcoCred – Eco-Friendly Loan Eligibility & Default Prediction

> 🧪 **Status**: Under Development  
> 📦 **Version**: v0.1  
> 🔗 **Live Demo**: _Coming Soon_

---

## 📝 Project Overview

**EcoCred** is an intelligent loan approval system that combines financial assessment with eco-friendly behavior. Users are evaluated not only on traditional financial metrics (income, credit score, loan amount) but also on their environmental footprint, such as electricity usage and vehicle fuel type.  

By integrating **ML models, OCR, and a web interface**, EcoCred encourages sustainability while predicting loan approval and default risk accurately.

---

## 🎯 Objectives

- Predict loan default risk using ML algorithms.  
- Calculate an **eco score** from electricity bills and vehicle type.  
- Use **OCR** to automatically extract key data from uploaded documents.  
- Provide personalized loan recommendations.  
- Build a web-based interface for real-world simulation.

---

## 🔄 Methodology

### 📊 Data Sources
- Lending Club dataset (cleaned & balanced)  
- Simulated eco-friendly data (electricity consumption, fuel type)  
- Sample electricity bill & RC images  

### ⚙️ Machine Learning Models
- **LightGBM** (primary model)  
- Logistic Regression & Random Forest (for experimentation)  

### 🧠 Input Features
- Income  
- Loan Amount  
- Electricity Units (from OCR)  
- Vehicle Type (from RC)  
- Eco Score (calculated)  

### 🔍 OCR & Image Preprocessing
- **Pytesseract** extracts:  
  - ⚡ kWh units from electricity bills  
  - ⛽ Fuel type from RC  
- Image preprocessing includes: grayscale, thresholding, and noise removal  

---

## 🧩 Modules

| Module                 | Description                                                  |
|------------------------|--------------------------------------------------------------|
| 👤 User Input           | Collects financial data, electricity bill & RC images       |
| 🧾 OCR                  | Extracts electricity usage & fuel type from images          |
| 🌍 Eco Scoring          | Calculates eco score based on user’s green behavior         |
| 🤖 ML Prediction        | Predicts loan approval (0 = Rejected, 1 = Approved)         |
| 📊 Result Display       | Displays approval, eco score, and recommendations           |

---

## 🧰 Tech Stack

| Category       | Tools/Libraries                               |
|----------------|-----------------------------------------------|
| Language       | Python                                        |
| ML Libraries   | `sklearn`, `pandas`, `numpy`, `lightgbm`     |
| OCR            | `pytesseract`, `opencv`                       |
| Web Framework  | `Flask` (backend), HTML/CSS/JS (frontend)    |
| Database       | `MongoDB` (planned for user history tracking)|
| Version Control| Git, GitHub                                   |

---

## 💡 Features

✅ Trained ML model for loan prediction  
✅ Eco score calculation via OCR  
✅ File upload interface for electricity bills & RC  
✅ Personalized loan recommendations  
🕓 MongoDB integration (in progress)  
🕓 PDF generation for loan approval (planned)  
🕓 User login and history-based scoring (planned)  

---

## 📈 Sample Inputs

| Feature            | Example Values                      |
|--------------------|-------------------------------------|
| Monthly Income     | ₹65,000                             |
| Loan Amount        | ₹25,000                             |
| Electricity Units  | 220 kWh (from bill via OCR)         |
| Vehicle Type       | Petrol / Diesel / Electric (from RC)|
| Eco Score          | 0 – 100 (auto-calculated)           |

---

## 🚀 Future Enhancements

- Google login + personalized dashboard  
- Loan status tracker with MongoDB backend  
- Aadhaar-based verification  
- Deep learning-based OCR for improved accuracy  
- Automated loan approval letters (PDF)  
- EMI tracker & reminders  

---

## 📁 Project Structure

```bash
EcoCred/
│
├── app.py                  # Flask backend logic
├── templates/              # HTML frontend templates
├── static/                 # CSS, JS, images
├── notebooks/              # ML & OCR notebooks
├── backend/                # Model, utilities, and API scripts
├── frontend/               # React or web frontend code (if applicable)
├── .gitignore              # Excluded files
└── README.md               # Project documentation
