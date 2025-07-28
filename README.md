# 🌍 Greenhouse Gas Emission Prediction - ML Models (Week 2)

This project focuses on building regression models to predict **Greenhouse Gas (GHG) Emission Factors** using supply chain data for US commodities from 2016.

## 📁 Dataset
- Source: `SupplyChainEmissionFactorsforUSIndustriesCommodities.xlsx`
- Sheet used: `2016_Summary_Commodity`
- Filtered for: `carbon dioxide` entries only

## 🎯 Objective
To predict the **Supply Chain Emission Factors with Margins** using various quality and emission-related features.

---

## 📊 Features Used

| Feature | Description |
|--------|-------------|
| `Supply Chain Emission Factors without Margins` | Raw emission factor |
| `Margins of Supply Chain Emission Factors` | Additional margin added |
| `DQ ReliabilityScore` | Data quality: reliability |
| `DQ TemporalCorrelation` | Data quality: time-based |
| `DQ GeographicalCorrelation` | Data quality: location-based |
| `DQ TechnologicalCorrelation` | Data quality: tech-based |
| `DQ DataCollection` | How data was collected |

---

## 🤖 Models Built

1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**
4. ✅ **Tuned Random Forest** (via GridSearchCV)

---

## 🧪 Evaluation Metrics

- **RMSE** - Root Mean Squared Error
- **MAE** - Mean Absolute Error
- **R² Score** - Coefficient of Determination

All models were evaluated on a test split (20%).

---

## 📈 Visualizations

- 📊 **Residual Plot** – to analyze prediction errors
- 🌲 **Feature Importance** – from the best model
- 📊 **Model Comparison** – RMSE, MAE, R² across models

---

## 🏆 Best Model

- **Random Forest Regressor (Tuned)** was the best performer.
- Saved as: `best_random_forest_model.pkl` using `joblib`

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python your_notebook_or_script.py
