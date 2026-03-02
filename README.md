\# Actuarial Insurance Claim Modeling Project



\## Overview



This project implements a complete actuarial modeling framework for estimating expected insurance claim cost using the traditional frequency–severity decomposition approach, combined with modern machine learning techniques.



The objective is to model:



\- Claim Frequency (number of claims)

\- Claim Severity (claim amount)

\- Expected Claim Cost



The study integrates statistical regression models and machine learning algorithms to evaluate predictive performance and actuarial applicability.



---



\## Project Structure



\- \*Chapter 1 – Data Cleaning \& Exploratory Data Analysis\*

&nbsp; - Data preprocessing

&nbsp; - Missing value handling

&nbsp; - Distribution analysis

&nbsp; - Class imbalance assessment



\- \*Chapter 2 – Frequency Modeling\*

&nbsp; - Poisson Regression

&nbsp; - Negative Binomial Regression

&nbsp; - Model comparison using AIC, BIC, RMSE, MAE



\- \*Chapter 3 – Severity Modeling\*

&nbsp; - Continuous loss modeling

&nbsp; - Parametric severity distribution fitting

&nbsp; - Model diagnostics and evaluation



\- \*Chapter 4 – Machine Learning Models\*

&nbsp; - Supervised classification models

&nbsp; - SMOTE for class imbalance handling

&nbsp; - Precision–Recall curve evaluation

&nbsp; - Model performance comparison



\- \*Chapter 5 – Expected Claim Cost Estimation\*

&nbsp; - Integration of selected frequency and severity models

&nbsp; - Expected loss estimation:

&nbsp; 

&nbsp;   Expected Claim Cost = Predicted Frequency × Predicted Severity



---



\## Datasets Used



\- \*Kaggle Motor Insurance Dataset\*

&nbsp; - Used for frequency modeling and machine learning classification.

&nbsp; - Contains policy-level claim occurrence data suitable for count and binary modeling.



\- \*Mendeley Insurance Claims Dataset\*

&nbsp; - Used for severity modeling.

&nbsp; - Contains continuous claim amount information appropriate for loss magnitude estimation.



---



\## Modeling Approach



\### Frequency (Count-Based Modeling)

\- Negative Binomial Regression selected due to overdispersion in claim counts.

\- Evaluated using:

&nbsp; - MSE

&nbsp; - MAE

&nbsp; - RMSE

&nbsp; - AIC

&nbsp; - BIC



\### Severity (Continuous Modeling)

\- Parametric regression applied to model positive claim amounts.

\- Appropriate for right-skewed loss distributions.



\### Machine Learning

\- Supervised classification models implemented.

\- SMOTE applied to address class imbalance.

\- Precision–Recall curves used instead of ROC curves due to imbalance considerations.



---



\## Key Insights



\- Claim data exhibits significant class imbalance.

\- Negative Binomial regression provided superior fit over Poisson due to overdispersion handling.

\- Precision–Recall evaluation provided more reliable performance assessment under imbalance.

\- Frequency–severity integration allows actuarially consistent estimation of expected loss.



---



\## Project structure



├── data/                      # Saved outputs (predictions, feature importance, coefficients)

├── notebooks/                 # Jupyter notebooks for each chapter

├── plots/                     # Core Python plots

├── README.md                  # Project documentation

└── requirements.txt   



---



\## Limitations



Frequency and severity models were estimated on separate but representative datasets.  

The expected claim cost was approximated using the product of average predicted frequency and average predicted severity, which may introduce approximation bias.



---



\## Author



Actuarial Science Student  

Focus: Statistical Modeling, Data Science, Insurance Analytics





---



\## Future Improvements



\- Unified dataset for joint frequency–severity modeling

\- Deployment-ready risk scoring pipeline 

\- SHAP-based interpretability analysis

