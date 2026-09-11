# 📞 Telecom Customer Churn Prediction

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to churn based on customer demographics, contract details, service usage, and billing information.

---


## 🚀 Project Overview

Customer churn is one of the biggest challenges faced by telecom companies. This project analyzes customer behavior and builds predictive models to identify customers at risk of leaving the service.

The final solution includes:

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training & Evaluation
- Feature Importance Analysis
- Interactive Streamlit Web Application

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Streamlit

---

## 📊 Key Insights

- Month-to-month contracts showed significantly higher churn rates.
- Customers using Fiber Optic services were more likely to churn.
- Customer tenure strongly influenced retention.
- Contract type was one of the most important predictive features.

---

## 🤖 Models Evaluated

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

**Best model: TODO (e.g. Random Forest)**

| Metric | Value |
|---|---|
| Accuracy | ~82% |
| Precision (Churn class) | TODO |
| **Recall (Churn class)** | TODO |
| F1 Score (Churn class) | TODO |

**Why recall matters most here:** on a churn dataset, most customers usually don't churn, so accuracy alone can be misleading — a model that just predicts "no churn" for everyone could still score high on accuracy while catching zero actual churners. Missing an actual churner (a false negative) costs the business a lost customer with no warning; a false alarm just costs one unnecessary retention offer. TODO: state your actual recall number here and one sentence on whether it's high enough to be useful for a real retention campaign, or what you'd do next to improve it (e.g. class-weighting, SMOTE, adjusting the decision threshold).

To get the real numbers, run this against your trained model and test set:
```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
```
Pull the "Churn" row's precision/recall/F1 into the table above.

---

## 🔎 Confusion Matrix

![Confusion Matrix](screenshots/confusion_matrix.png)

Shows how many actual churners the model caught vs. missed — the false negative count (bottom-left cell) is the number that matters most for this business problem.

Code to generate this image:
```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6,5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=['No Churn', 'Churn'],
    cmap='Blues', ax=ax
)
plt.title('Confusion Matrix — Telecom Churn')
plt.tight_layout()
plt.savefig('screenshots/confusion_matrix.png', dpi=150)
```

---

## 🖥 Streamlit Application

The application allows users to:

- Enter customer details
- Generate churn predictions instantly
- Assess churn risk in real time

---

## 📂 Repository Structure

```
telecom-customer-churn-prediction/
├── app.py
├── main.py
├── TelecomCustomerChurnPrediction.ipynb
├── random_forest_model.joblib
├── Telco_Customer_Churn.csv
├── requirements.txt
└── screenshots/
    └── confusion_matrix.png
```

---

## 👨‍💻 Author

Siddhant Varma

Data Science & Machine Learning Enthusiast
