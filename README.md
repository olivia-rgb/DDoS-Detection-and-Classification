🛡️ DDoS Detection using Machine Learning
📖 Introduction

Distributed Denial of Service (DDoS) attacks remain a critical cybersecurity threat that can overwhelm and disrupt online services. Effective detection and mitigation are necessary to ensure the reliability of networks.

This project focuses on classifying DDoS attacks using the CIC-IDS 2024 dataset containing 317 features. The dataset provides a comprehensive set of traffic statistics that allow machine learning algorithms to detect malicious patterns.

We implement and compare multiple machine learning models, including:

Random Forest

Logistic Regression

XGBoost

The models are trained and evaluated systematically through preprocessing, feature analysis, training, evaluation, and comparison to identify the most effective approach.

📑 Table of Contents

Importing Libraries

Data Pre-processing

Data Exploring

Data Splitting

Model Training

Random Forest

Logistic Regression

XGBoost

Model Evaluation

Accuracy

F1 Score

Recall

Precision

Confusion Matrix

Feature Importance

Model Comparison

🔹 Chapters Overview
1. Importing Libraries

We use Python libraries for data handling, visualization, training, and evaluation:

pandas, numpy → data manipulation

matplotlib, seaborn → visualization

scikit-learn → preprocessing, ML models, evaluation

xgboost → gradient boosting

2. Data Pre-processing

Steps include:

Handling missing values

Encoding categorical features (e.g., attack labels)

Feature scaling using RobustScaler (handles outliers well)

Ensuring all 317 features are included in training

3. Data Exploring

We generate descriptive statistics and visualizations to:

Examine feature distributions

Analyze class balance (Benign vs DDoS)

Identify correlations between features

4. Data Splitting

Data is split into train (80%) and test (20%), stratified by attack class.
This ensures fair evaluation across all categories.

5. Model Training

We train three models:

Random Forest → ensemble of decision trees

Logistic Regression → baseline classifier

XGBoost → advanced boosting model optimized for large feature sets like this dataset

6. Model Evaluation

We evaluate performance using:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix (visual insight into misclassifications)

We also analyze feature importance (from Random Forest & XGBoost) to highlight which of the 317 features are most relevant for detecting attacks.

7. Model Comparison

The performance of all models is compared side by side.

XGBoost typically performs best for high-dimensional datasets like CIC-IDS 2024.

Random Forest also gives strong results and feature interpretability.

Logistic Regression acts as a baseline.

✅ Conclusion

This project demonstrates the effectiveness of machine learning in detecting DDoS attacks using a high-dimensional dataset with 317 features.

Key outcomes:

XGBoost shows the strongest performance

Feature importance analysis highlights critical attributes in network traffic

A reproducible workflow for future experiments on cybersecurity datasets
