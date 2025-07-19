# 🧹 Data Cleaning & Visualization Notebook

This project provides a complete Google Colab-based framework for handling missing values, removing outliers, and visualizing datasets using Python.

---

## 📌 Features

- ✅ Upload and preview CSV data
- ✅ Fill missing values (mean for numeric, mode for categorical)
- ✅ Remove outliers using the IQR method (safely, per column)
- ✅ Visualize data (histograms, boxplots, correlation heatmaps)
- ✅ Download the cleaned dataset

---

## 📂 Files

- `Data_Cleaning_and_Visualization.ipynb` – Google Colab notebook with all code
- `cleaned_no_outliers.csv` – Sample cleaned dataset (generated after running the notebook)

---

## 🚀 How to Use

1. **Open in Google Colab**  
   Click below to launch the notebook:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](URL_TO_YOUR_NOTEBOOK)

2. **Upload your dataset**  
   - The notebook will prompt you to upload a `.csv` file

3. **Run all cells**  
   - Missing values will be handled automatically
   - Outliers will be removed column-wise using IQR
   - Visualizations will help you understand data distribution

4. **Download cleaned CSV**  
   - Use the final cell to download the processed file

---

## 🧠 Libraries Used

- `pandas` – data manipulation  
- `matplotlib` & `seaborn` – data visualization  
- `google.colab` – upload & download files in Colab

---

## 🧪 Example Visualizations

- 🔍 Histograms of numeric features  
- 📦 Boxplots for outlier inspection  
- 🔥 Correlation heatmap

---

## 🛠️ Customize It

- Replace IQR logic with Z-Score if needed
- Add one-hot encoding or label encoding for ML prep
- Integrate with a Streamlit or Flask app

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

---

## 📃 License

This project is licensed under the [MIT License](LICENSE).

---

## 💡 Author

**Akash K** – [LinkedIn](https://www.linkedin.com/in/akash-k-a12842327)

---

## 🌌 Made with ❤️ in Python
