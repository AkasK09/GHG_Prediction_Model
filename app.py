import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(layout="wide")
st.title("📊 Supply Chain Emission Factor Model Comparison")

# Upload Excel file
uploaded_file = st.file_uploader("📁 Upload Excel File", type=["xlsx"])

if uploaded_file:
    try:
        # Load data
        df = pd.read_excel(uploaded_file, sheet_name="2016_Summary_Commodity")
        substances = df['Substance'].dropna().unique().tolist()
        selected_substance = st.selectbox("🔍 Select Substance", substances)

        df = df[df['Substance'] == selected_substance]

        # Features and target
        features = [
            'Supply Chain Emission Factors without Margins',
            'Margins of Supply Chain Emission Factors',
            'DQ ReliabilityScore of Factors without Margins',
            'DQ TemporalCorrelation of Factors without Margins',
            'DQ GeographicalCorrelation of Factors without Margins',
            'DQ TechnologicalCorrelation of Factors without Margins',
            'DQ DataCollection of Factors without Margins'
        ]
        target = 'Supply Chain Emission Factors with Margins'

        if df[features + [target]].isnull().any().any():
            st.warning("⚠️ There are missing values in your data. Please clean the dataset.")
        else:
            X = df[features]
            y = df[target]

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train models
            lr = LinearRegression()
            dt = DecisionTreeRegressor(random_state=42)
            rf = RandomForestRegressor(random_state=42)

            lr.fit(X_train, y_train)
            dt.fit(X_train, y_train)
            rf.fit(X_train, y_train)

            # Evaluation function
            def evaluate(name, model):
                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.markdown(f"**{name}**  \nRMSE: `{rmse:.4f}`, MAE: `{mae:.4f}`, R²: `{r2:.4f}`")
                return rmse, mae, r2

            # Evaluate models
            st.subheader("📈 Model Evaluation")
            results = {}
            results['Linear Regression'] = evaluate("Linear Regression", lr)
            results['Decision Tree'] = evaluate("Decision Tree", dt)
            results['Random Forest'] = evaluate("Random Forest", rf)

            # Grid Search on RF
            st.subheader("🛠️ Hyperparameter Tuning (Random Forest)")
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [None, 10],
                'min_samples_split': [2, 5]
            }
            gs = GridSearchCV(
                RandomForestRegressor(random_state=42),
                param_grid,
                cv=3,
                scoring='r2',
                n_jobs=-1
            )
            gs.fit(X_train, y_train)

            st.write("**Best Parameters**:", gs.best_params_)
            best_rf = gs.best_estimator_
            results['Tuned Random Forest'] = evaluate("Tuned Random Forest", best_rf)

            # Residual Plot
            st.subheader("📉 Residuals")
            residuals = y_test - best_rf.predict(X_test)
            fig1 = px.scatter(
                x=y_test,
                y=residuals,
                labels={'x': "Actual", 'y': "Residuals"},
                title="Residual Plot",
                trendline="lowess"
            )
            st.plotly_chart(fig1, use_container_width=True)

            # Feature Importances
            st.subheader("🌟 Feature Importances")
            importances = pd.Series(best_rf.feature_importances_, index=features)
            fig2 = px.bar(
                importances.sort_values(),
                orientation='h',
                title="Feature Importances",
                labels={'value': 'Importance', 'index': 'Feature'}
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Model Comparison
            st.subheader("📊 Model Comparison")
            model_names = list(results.keys())
            rmse_vals = [v[0] for v in results.values()]
            mae_vals = [v[1] for v in results.values()]
            r2_vals = [v[2] for v in results.values()]

            fig3 = make_subplots(rows=1, cols=3, subplot_titles=("RMSE", "MAE", "R² Score"))

            fig3.add_trace(go.Bar(x=model_names, y=rmse_vals, name="RMSE"), row=1, col=1)
            fig3.add_trace(go.Bar(x=model_names, y=mae_vals, name="MAE"), row=1, col=2)
            fig3.add_trace(go.Bar(x=model_names, y=r2_vals, name="R²"), row=1, col=3)

            fig3.update_layout(showlegend=False, height=400, width=1000)
            st.plotly_chart(fig3, use_container_width=True)

            # Save best model
            joblib.dump(best_rf, "best_random_forest_model.pkl")
            st.success("✅ Best model saved as `best_random_forest_model.pkl`")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
