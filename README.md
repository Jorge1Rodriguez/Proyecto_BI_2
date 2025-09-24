# Home Credit Default Risk Prediction

## 📋 Project Overview

This project implements a machine learning pipeline to predict loan default risk as a binary classification problem. The goal is to predict whether loan applicants will have difficulties repaying their loans using various applicant features. The evaluation metric used is the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

---

## 🎯 Problem Statement

The task is to predict whether a person applying for a home credit will experience payment difficulties.

- **1**: The client will likely default (payment delay in installments)
- **0**: The client will not have payment difficulties

---

## 📁 Project Structure

```
project2/
├── dataset/
│   ├── application_train_aai.csv       # Training dataset (246,008 samples, 122 features)
│   ├── application_test_aai.csv        # Test dataset with model predictions (61,503 samples)
│   ├── application_test.csv            # Original copy of test dataset without model predictions (for comparison purposes)
│   ├── HomeCredit_columns_description.csv # Feature descriptions
├── model/
│   └── random_forest_model.pkl              # Serialized best Random Forest model
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_utils.py
│   └── preprocessing.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_data_utils.py
│   └── test_preprocessing.py
├── Project 02.ipynb                    # Main Jupyter Notebook
├── requirements.txt
├── README.md
└── venv/
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Installation

1. Clone or download the project:

```bash
git clone <repository-url>
cd project2
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Open the Jupyter Notebook and execute each cell in order


---

## ⚙️ Model Usage and Training

- **Pre-trained model available:** The project includes a trained Random Forest model stored in `models/random_forest_model.pkl` ready for inference and prediction.
- **Retrain if desired:** Users can customize and retrain the model by adjusting hyperparameters in the RandomizedSearchCV section of the notebook (Section 3.7).

---

## 📊 Dataset Information

### Training Dataset
- `application_train_aai.csv` contains 246,008 samples with 122 features
- Imbalanced classes with about 8% positive (default) and 92% negative

### Test Dataset
- `application_test_aai.csv` contains 61,503 samples and includes model predictions in the TARGET column
- `application_test.csv` is an original copy of the test dataset without predictions, used for comparison purposes

---

## 🎯 Evaluation Metric

The model's performance is measured using ROC AUC, which quantifies the model's ability to distinguish between the two classes regardless of classification thresholds. ROC AUC ranges from 0.5 (random guessing) to 1.0 (perfect classification).

---

## 🧪 Testing

Run unit tests to validate the project implementation:

```bash
pytest tests/
```

---

## 📝 Additional Notes

- The data preprocessing pipeline includes outlier correction, encoding of categorical variables, missing value imputation, and feature scaling to prepare the data for modeling.
- Care has been taken to avoid data leakage by fitting transformers only on training data.
- The dataset size requires mindful resource and memory management.
- Class imbalance is a factor to consider for evaluation and potential improvement.
- Training time for the Random Forest and hyperparameter tuning can be substantial.

---

**The model is ready for immediate use, and the test prediction file is included.**