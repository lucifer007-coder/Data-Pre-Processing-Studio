# 📊 Streamlit Data Preprocessing App

An interactive **Streamlit web application** that simplifies data preprocessing for analytics and machine learning.
This app enables users to upload CSV files, apply multiple preprocessing steps through an intuitive UI, preview results at each stage, and download the final cleaned dataset.

The app is designed to address **common pain points in data preprocessing** such as missing data, duplicates, outliers, scaling, and imbalanced datasets—all without writing a single line of code.

---

## 🚀 Features

### ✅ 1. Handling Missing Data

* Impute with **Mean / Median / Mode / Constant** value
* Drop rows or columns with missing values

### ✅ 2. Data Inconsistencies

* Standardize date formats
* Normalize text (lowercasing, trimming spaces)
* Handle unit conversions (if specified)

### ✅ 3. Noisy Data & Outliers

* Detect and filter outliers using **IQR** or **Z-score**
* Options to **Cap, Remove, or Transform** outliers

### ✅ 4. Data Duplication

* Identify and remove duplicate rows
* Choose specific columns to check duplicates against

### ✅ 5. Categorical Data Handling

* One-Hot Encoding
* Label Encoding

### ✅ 6. Feature Scaling & Normalization

* Apply **Standardization** (z-score scaling)
* Apply **Min-Max Scaling**

### ✅ 7. Imbalanced Data (for classification datasets)

* Random Oversampling
* Random Undersampling

### ✅ 8. Pipeline Automation

* Chain multiple preprocessing steps
* Preview results before applying changes
* Automated, iterative preprocessing with **undo safety**

### ✅ 9. Dashboard & Reporting

* Before/After dataset statistics
* Distribution plots, missing value counts, and summary tables
* Change log of applied preprocessing steps

### ✅ 10. Export Final Dataset

* Download the **cleaned CSV file**

---

## 🛠️ Tech Stack

* **Frontend/UI:** [Streamlit](https://streamlit.io/)
* **Data Handling:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
* **Machine Learning Preprocessing:** [Scikit-learn](https://scikit-learn.org/)
* **Visualization:** Streamlit charts, Matplotlib, Altair

---

## 📂 Project Structure

```
📦 data-preprocessing-app
 ┣ 📜 app.py               # Main Streamlit app script
 ┣ 📜 README.md            # Project documentation (this file)
 ┗ 📂 sample_data          # (Optional) Example CSV files for testing
```

---

## ▶️ Installation & Usage

### 1️⃣ Clone this repository

```bash
git clone https://github.com/your-username/data-preprocessing-app.git
cd data-preprocessing-app
```

### 2️⃣ Install dependencies

It’s recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, install manually:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib altair
```

### 3️⃣ Run the app

```bash
streamlit run app.py
```

### 4️⃣ Open in browser

The app will run locally at:
👉 [http://localhost:8501](http://localhost:8501)

---

## 📊 Example Workflow

1. **Upload a CSV file**
2. Select preprocessing steps from the sidebar:

   * Handle missing data
   * Normalize text
   * Remove duplicates
   * Encode categorical columns
   * Scale features
   * Balance dataset
3. **Preview changes** at each step (dataframe, histograms, summary stats)
4. Review the **dashboard report** of applied steps
5. **Download** the final cleaned dataset as CSV

---

## 🧩 Future Enhancements

* Advanced outlier detection (Isolation Forest, DBSCAN)
* Automated feature engineering
* Integration with ML pipelines (train/test split, feature selection)
* Export preprocessing pipeline as a reusable script

---

## 🙌 Acknowledgements

* Inspired by real-world challenges in **data preprocessing**
* Built with ❤️ using **Streamlit**

---

Would you like me to also create a **`requirements.txt` file** alongside this README so that users can install everything with one command?
