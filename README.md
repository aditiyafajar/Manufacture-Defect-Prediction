
[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

# Predictive Modeling and Optimization of Manufacturing Defects
This project aims to analyze manufacturing processes and predict defect occurrence using machine learning. By identifying key factors contributing to defects, the project provides actionable insights to improve production efficiency and product quality.

## Table of Contents
1. [Background](#background)
2. [Dataset Information](#dataset-information)
3. [Data Preprocessing and Analysis](#data-preprocessing-and-analysis)
4. [Machine Learning Models](#machine-learning-models)
5. [Results and Insights](#results-and-insights)
6. [Conclusions and Recommendations](#conclusions-and-recommendations)
7. [How to Use](#how-to-use)
8. [Technologies Used](#technologies-used)
9. [Contact](#contact)

---

## Background

Manufacturing defects are a persistent problem, increasing production costs, downtime, and reducing customer trust. This project focuses on:
- Evaluating production variables to identify defect-causing factors.
- Developing a predictive model to detect high-risk batches requiring quality inspection.

**Business Objectives**:
- Reduce defect rates by identifying contributing factors.
- Enhance production predictability using machine learning models.

---

## Dataset Information

- **Dataset Source**: [Kaggle - Predicting Manufacturing Defects](https://www.kaggle.com/datasets/rabieelkharoua/predicting-manufacturing-defects-dataset/data)
- **Size**: 3,240 rows, 17 columns
- **Data Type**: Numerical (integer and float)
- **Target Variable**: `DefectStatus` (1 = defective, 0 = non-defective)

**Features**:
- Key variables include `EnergyConsumption`, `AdditiveMaterialCost`, `MaintenanceHours`, and `DefectRate`.

---

## Data Preprocessing and Analysis

1. **Data Cleaning**:
   - No missing values or duplicate rows.
   - Imbalanced `DefectStatus` distribution addressed in modeling.

2. **Exploratory Data Analysis (EDA)**:
   - Distribution analysis indicates clean, consistent features.
   - Key insights:
     - High maintenance hours and low supplier quality lead to increased defects.
     - Production volume slightly decreases with higher defect levels.

3. **Feature Engineering**:
   - Dropped redundant features (`SupplierQuality`, `WorkerProductivity`) based on correlation and VIF analysis.
   - Selected 13 features after refinement.

---

## Machine Learning Models

Several models were trained and evaluated:
- **Random Forest Classifier** (Tuned): Best performance with balanced accuracy across both classes.
- **XGBoost**: Slightly weaker than Random Forest in recognizing the "No Defect" class.
- **Logistic Regression**: High recall for the defective class but poor for non-defective.
- **KNN and SVM**: Poor performance in recognizing non-defective products.

**Best Model**:
- **Random Forest (Tuned)**:
  - Parameters: `{'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_depth': 10, 'class_weight': 'balanced', 'bootstrap': False}`
  - Performance:
    - Recall for "No Defect" improved from 64% to 73%.
    - Macro Average Recall: Increased from 78% to 82%.

---

## Results and Insights

1. **Top Features Influencing Defects**:
   - `MaintenanceHours`, `DefectRate`, `ProductionVolume`

2. **Model Insights**:
   - High maintenance hours combined with low supplier quality indicate significant defect risks.
   - Prioritizing batches with historical high defect rates can improve defect predictability.

3. **AUC-ROC Scores**:
   - Random Forest: 0.80
   - XGBoost: 0.81
   - Logistic Regression: 0.77

---

## Conclusions and Recommendations

### Conclusions:
- Random Forest and XGBoost are the most suitable models for predicting manufacturing defects.
- MaintenanceHours and DefectRate are the most critical features influencing defect rates.

### Recommendations:
1. Implement preventive maintenance with well-planned schedules.
2. Perform additional inspections on high-risk batches.
3. Optimize production planning to ensure manageable capacity limits.
4. Strengthen supplier audits to ensure higher quality standards.

---

## How to Use

### Prerequisites:
- Python 3.8 or above
- Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost` 

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/aditiyafajar/manufacturing-defect-prediction.git
2. Install dependencies
    ```bash
    pip install -r requirements
3. Run Jupyter Notebook:
    ```bash
    jupyter notebook \Defect Prediction-Aditiya Fajar.ipynb
---
### Technology used: 
- Languages: Python
- Libraries: Scikit-learn, XGBoost, Matplotlib, Seaborn, Pandas, Numpy
- Tools: Jupyter Notebook, Kaggle
---
### Contact: 
**Author: Aditiya Fajar**
- Email: aditiyafajar68@gmail.com
- LinkedIn: https://www.linkedin.com/in/aditiyafajar
- GitHub: https://github.com/aditiyafajar

---
Thank you for exploring this project! Feedback and contributions are welcome.
Feel free to customize it further to match your preferences! Let me know if you need help integrating this into your GitHub repository.


