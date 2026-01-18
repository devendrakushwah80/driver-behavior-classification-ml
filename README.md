# 🚗 Driver Behavior Classification

## 📌 Project Overview

This project focuses on analyzing and classifying **driver behavior** using machine learning techniques. The notebook walks through a complete **end-to-end ML workflow** starting from data loading and exploration to model training, validation, and evaluation using pipelines and cross‑validation.

The goal is to build a robust classification model that can predict driver behavior patterns based on numerical features.

---

## 📂 Dataset

* **File used:** `Driver_Behavior.csv`
* **Type:** Structured tabular dataset
* **Target:** Driver behavior class (categorical)
* **Features:** Numerical driving-related attributes (scaled during preprocessing)

> The dataset is loaded using **Pandas**, and basic inspection is performed using `head()`, `tail()`, `shape()`, and `info()`.

---

## 🔍 Exploratory Data Analysis (EDA)

The notebook includes:

* Dataset shape and structure inspection
* Null value checks
* Statistical summary
* Feature distribution visualization using **Matplotlib** and **Seaborn**

These steps help understand feature ranges and prepare the data for modeling.

---

## ⚙️ Data Preprocessing

The preprocessing pipeline includes:

* **Train-test split** using `train_test_split`
* **Feature scaling** with `StandardScaler`
* Clean separation of features (`X`) and target (`y`)

All preprocessing steps are handled using **Scikit‑learn Pipelines**, ensuring clean and leak‑free training.

---

## 🧠 Machine Learning Models

The notebook experiments with:

### ✅ K‑Nearest Neighbors (KNN)

* Implemented using `KNeighborsClassifier`
* Hyperparameter tuning using `GridSearchCV`
* Cross‑validation to measure generalization

### ✅ Pipeline Usage

```text
Pipeline → Scaling → Model
```

Using pipelines ensures reproducibility and consistency between training and testing data.

---

## 📊 Model Evaluation

The model is evaluated using:

* **Accuracy score** on test data
* **Cross‑validation mean & standard deviation**
* Comparison of train vs test performance

This ensures the model is neither overfitting nor underfitting.

---

## 🧪 Cross‑Validation

* K‑Fold Cross‑Validation applied
* Mean and standard deviation of scores analyzed
* Helps in selecting optimal hyperparameters

---

## 📈 Results

* Scaled features significantly improve model performance
* KNN performs well after tuning `n_neighbors`
* Pipeline‑based approach simplifies experimentation

---

## 🛠️ Tech Stack

* **Python**
* **Pandas & NumPy** – Data handling
* **Matplotlib & Seaborn** – Visualization
* **Scikit‑learn** – ML models, pipelines, and evaluation

---

## 📁 Repository Structure

```text
├── Driver_Behavior.csv
├── driver_behavior_classification.ipynb
├── README.md
```

---

## 🚀 How to Run

1. Clone the repository
2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Open the notebook

```bash
jupyter notebook driver_behavior_classification.ipynb
```

---

## 📌 Key Learning Outcomes

* Proper ML pipeline design
* Importance of feature scaling
* Hyperparameter tuning with GridSearchCV
* Cross‑validation for reliable evaluation

---

## ✨ Future Improvements

* Try advanced models (Random Forest, XGBoost)
* Add confusion matrix & classification report
* Perform feature importance analysis

---

### 👤 Author

**Devendra Kushwah**
