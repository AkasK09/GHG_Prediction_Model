import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="GHG Emission Prediction", page_icon="🌍", layout="wide")

st.title("🌍 GHG Emission Prediction App")
st.write("Upload a dataset and predict Greenhouse Gas (GHG) Emissions using Machine Learning.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload Excel dataset", type=["xlsx"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_excel(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Automatically select last column as target
    target_col = df.columns[-1]
    feature_cols = df.columns[:-1]

    st.success(f"✅ Target column detected: **{target_col}**")
    st.write(f"✅ Feature columns: {list(feature_cols)}")

    # Prepare data
    X = df[feature_cols]
    y = df[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("📈 Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Mean Squared Error", f"{mse:.2f}")
    col2.metric("R² Score", f"{r2:.2f}")

    # Prediction section
    st.subheader("🔮 Predict Emission for Custom Input")

    input_data = []
    for col in feature_cols:
        val = st.number_input(
            f"Enter {col}",
            float(df[col].min()),
            float(df[col].max()),
            float(df[col].mean())
        )
        input_data.append(val)

    if st.button("Predict Emission"):
        prediction = model.predict([input_data])
        st.success(f"🌱 Predicted {target_col}: **{prediction[0]:.2f}**")

else:
    st.info("👆 Please upload an Excel dataset to continue.")
