# EcoCred 🌱  

EcoCred is a Machine Learning–powered project designed to evaluate an individual's eco-friendliness and provide personalized scheme recommendations, benefits, and insights.  

---

## 🔑 Project Overview
- **Goal**: Promote eco-conscious behavior by rewarding individuals through interest reductions, offers, and schemes.  
- **Dataset**: ~2.2M rows, 150+ features (subset sampling for efficiency).  
- **Core Features**:
  - ⚡ OCR Extraction → Extract details from electricity bills (kWh usage) & RC images (fuel type, EV check).  
  - 📊 ML Prediction → LightGBM model trained with balanced dataset (25k eco-friendly & 25k non-eco-friendly).  
  - 🌍 Eco Score → Derived from energy usage, transport choices, and lifestyle data.  
  - 🎯 Scheme Recommendation → Suggests benefits/schemes based on eco score.  
  - 🤖 Chatbot → (In-progress) to answer user queries.  
  - 🔍 SHAP → Explainable AI for feature importance & decision transparency.  

---

## 📂 Project Structure
### Current (Improved) Version → `fresh-project` branch
- ✅ Uses **LightGBM** for better performance on large datasets.  
- ✅ Feature selection applied (30 columns kept, 10–12 user inputs + derived features).  
- ✅ Balanced dataset sampling for improved accuracy & precision.  
- ✅ Streamlit app for demo & UI testing.  

### Old (Initial) Version → `main` branch
- ❌ Used **Random Forest** with only 10 features & smaller dataset.  
- ❌ Predictions were less accurate due to limited scope.  
- ❌ Kept for reference to show learning & iteration.  

---

## 🚀 Tech Stack
- **ML Models**: Random Forest (old), LightGBM (current)  
- **Python Libraries**: scikit-learn, lightgbm, shap, pandas, numpy  
- **Frontend**: Streamlit  
- **Other Features**: OCR (pytesseract, OpenCV), chatbot integration  

---

## 📊 Results
- Improved accuracy & precision with LightGBM after balancing dataset.  
- Clear interpretability with SHAP values.  
- Scalable for large datasets with 2M+ records.  

---

## 🔮 Future Scope
- Complete chatbot integration.  
- Expand scheme recommendation engine.  
- Deploy full Streamlit app with user authentication.  

---

## 👨‍💻 Author
Bharani Kumar  
