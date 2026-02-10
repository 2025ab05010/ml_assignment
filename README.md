📌 Breast Cancer Classification using Machine Learning and Streamlit
a) Problem Statement

The objective of this project is to implement multiple machine learning classification models on a real-world healthcare dataset and deploy them through an interactive Streamlit web application.

The system allows users to upload test data, select different trained models, and view evaluation metrics and confusion matrices for comparison.

The project demonstrates the complete end-to-end machine learning workflow including:

Data preprocessing

Model training

Model evaluation

Web UI development

Cloud deployment

b) Dataset Description

The dataset used is the Breast Cancer Wisconsin Diagnostic Dataset, obtained from Kaggle.

Key details:

Total samples: 569

Total features: 30 numerical features

Target variable: diagnosis

0 → Benign

1 → Malignant

No missing values after cleaning

Columns such as id and Unnamed: 32 were removed

The dataset satisfies the assignment constraints of minimum 12 features and 500 instances.

c) Models Used and Performance Comparison

Six machine-learning classification models were implemented and evaluated on the same dataset:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors

Naive Bayes (Gaussian)

Random Forest (Ensemble)

XGBoost (Ensemble)

📊 Comparison Table
ML Model	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.9649	0.9960	0.9750	0.9286	0.9512	0.9245
Decision Tree	0.9298	0.9246	0.9048	0.9048	0.9048	0.8492
KNN	0.9561	0.9825	0.9744	0.9048	0.9383	0.9058
Naive Bayes	0.9211	0.9891	0.9231	0.8571	0.8889	0.8292
Random Forest (Ensemble)	0.9737	0.9944	1.0000	0.9286	0.9630	0.9442
XGBoost (Ensemble)	0.9649	0.9940	1.0000	0.9048	0.9500	0.9258
d) Observations on Model Performance
ML Model	Observation
Logistic Regression	Achieved very high AUC and MCC due to near-linear separability of the features, making it a strong baseline model.
Decision Tree	Performed reasonably well but showed lower AUC, likely due to overfitting tendencies on training data.
KNN	Benefited significantly from feature scaling; showed high precision but slightly reduced recall.
Naive Bayes	Computationally efficient but its independence assumption limited recall compared to ensemble methods.
Random Forest	Best overall performer with perfect precision and the highest MCC, indicating excellent classification stability.
XGBoost	Delivered highly competitive results; strong accuracy and AUC but marginally lower recall than Random Forest.