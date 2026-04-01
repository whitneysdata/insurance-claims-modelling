!\[Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square\&logo=python)









!\[Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square\&logo=jupyter)









!\[Statsmodels](https://img.shields.io/badge/Statsmodels-GLM-lightblue?style=flat-square)









!\[XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red?style=flat-square)









!\[License](https://img.shields.io/badge/License-MIT-green?style=flat-square)









!\[Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)







\---



\# Insurance Claims Modelling

\## Frequency, Severity and Pure Premium Estimation

\### A Comparative Study of GLMs and Machine Learning



> \*\*Author:\*\* Whitney Kemuma

> \*\*Degree:\*\* BSc Actuarial Science

> \*\*Applying For:\*\* Research Master's in \[Data Science / Risk Analysis / Statistics / Machine Learning]

> \*\*Dataset:\*\* freMTPL2 — French Motor Third Party Liability Insurance (678,013 policies)

> \*\*Tools:\*\* Python, Statsmodels, Scikit-learn, XGBoost



\---



\## Abstract



This project builds and evaluates a full frequency-severity claims

modelling pipeline using the freMTPL2 French Motor Third Party

Liability dataset. Claim frequency is modelled using Poisson and

Negative Binomial regression. Claim severity is modelled using Gamma

and Lognormal regression. Machine learning models; Random Forest and

XGBoost are trained alongside GLMs and compared using RMSE, MAE and

R-squared. All frequency and severity predictions are combined into a

pure premium estimate, the expected claim cost per unit of exposure.



\---



\## Research Questions



| | Question |

|--|----------|

| \*\*RQ1\*\* | Does overdispersion in claim counts justify Negative Binomial over Poisson regression? |

| \*\*RQ2\*\* | Between Gamma and Lognormal, which better fits claim severity in the freMTPL2 dataset? |

| \*\*RQ3\*\* | Do ML models achieve meaningfully lower prediction error than GLM baselines? |

| \*\*RQ4\*\* | Which features are the strongest predictors of claim frequency and severity? |



\---



\## Project Structure

insurance-claims-modelling/

│

├── data/

│   ├── freMTPL2freq.csv.zip         <- Raw frequency file (Kaggle)

│   ├── freMTPL2sev.csv              <- Raw severity file (Kaggle)

│   ├── full\_data.csv.gz             <- Merged cleaned dataset

│   ├── severity\_data.csv.gz         <- Claims-only subset

│   ├── X\_freq\_train/test.csv.gz     <- Frequency splits

│   ├── X\_sev\_train/test.csv.gz      <- Severity splits

│   ├── y\_freq\_train/test.csv.gz     <- Frequency targets

│   ├── y\_sev\_train/test.csv.gz      <- Severity targets

│   ├── exposure\_train/test.csv.gz   <- Exposure for GLM offset

│   └── pure\_premium\_results.csv     <- Final pure premium output

│

├── models/

│   ├── poisson.pkl                  <- Fitted Poisson GLM

│   ├── neg\_binomial.pkl             <- Fitted Negative Binomial GLM

│   ├── gamma.pkl                    <- Fitted Gamma GLM

│   ├── lognormal.pkl                <- Fitted Lognormal GLM

│   ├── lognormal\_sigma2.pkl         <- Bias correction term

│   ├── rf\_frequency.pkl             <- Random Forest (frequency)

│   ├── xgb\_frequency.pkl            <- XGBoost (frequency)

│   ├── rf\_severity.pkl              <- Random Forest (severity)

│   ├── xgb\_severity.pkl             <- XGBoost (severity)

│   └── .csv                        <- Model comparison tables

│

├── notebooks/

│   ├── Chapter\_0\_Research\_Foundation.ipynb

│   ├── Chapter\_1\_Data.ipynb

│   ├── Chapter\_2\_EDA.ipynb

│   ├── Chapter\_3\_GLM\_Modelling.ipynb

│   ├── Chapter\_4\_ML\_Comparison.ipynb

│   └── Chapter\_5\_Evaluation.ipynb

│

├── plots/

│   ├── fig2\_.png                   <- EDA plots

│   ├── fig3\_.png                   <- GLM diagnostic plots

│   ├── fig4\_.png                   <- ML comparison plots

│   └── fig5\_\*.png                   <- Evaluation plots

│

├── src/

│   ├── data\_loader.py               <- Data loading functions

│   ├── eda\_utils.py                 <- EDA helper functions

│   ├── glm\_utils.py                 <- GLM loading and prediction

│   └── ml\_utils.py                  <- ML loading and prediction

│

├── README.md

└── requirements.txt

\---



\## Methodology



\### 1. Data Preparation

Two freMTPL2 files merged on policy ID. Severity file aggregated

to policy level by summing multiple claims per policy. Exposure

log transformed as GLM offset. Categorical features one-hot encoded

for ML models. 80/20 stratified train test split.



\### 2. Frequency Modelling

Poisson and Negative Binomial GLMs fitted with log(Exposure)

as offset. Overdispersion index = 1.083 confirmed in EDA.

AIC and BIC used for model selection.



\### 3. Severity Modelling

Gamma and Lognormal GLMs fitted on claims only subset.

Lognormal uses OLS on log(ClaimAmount) with bias correction.

AIC, BIC, RMSE and MAE used for model selection.



\### 4. Machine Learning Comparison

Random Forest and XGBoost trained for both frequency and severity.

Frequency modelled as binary classification (claim vs no claim).

Severity modelled as regression on log(ClaimAmount).



\### 5. Pure Premium

Pure Premium = E\[Frequency] x E\[Severity]

Four combinations evaluated:



| Combination | Frequency | Severity |

|-------------|-----------|----------|

| GLM x GLM | Negative Binomial | Gamma |

| GLM x ML | Negative Binomial | XGBoost |

| ML x GLM | XGBoost | Gamma |

| ML x ML | XGBoost | XGBoost |



\---



\## How to Run



\### 1. Clone the repository

```bash

git clone https://github.com/whitneysdata/insurance-claims-modelling.git

cd insurance-claims-modelling



2\. Install dependencies

pip install -r requirements.txt



3\. Download the dataset

Download from Kaggle — freMTPL2

and place both files in the data/ folder:

freMTPL2freq.csv.zip

freMTPL2sev.csv



4\. Run notebooks in order

Chapter\_0  <- Read only, research foundation

Chapter\_1  <- Run first — data preparation

Chapter\_2  <- EDA

Chapter\_3  <- GLM modelling (10-20 min)

Chapter\_4  <- ML comparison (10-20 min)

Chapter\_5  <- Evaluation and pure premium



Key Results

Full results available in models/final\_comparison.csv

and notebooks/Chapter\_5\_Evaluation.ipynb



| Combination | RMSE | MAE | R-squared |

|-------------|------|-----|-----------|

| GLM x GLM | 712.4929 | 208.7829 | -0.0109 |

| GLM x ML | 711.9917 | 166.7779 | -0.0095 |

| ML x GLM | 1023.1356 | 840.2523 | -1.0845 |

| ML x ML | 799.0027 | 519.4213 | -0.2713 |



\---



References

Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.

Cameron, A. C., \& Trivedi, P. K. (1986). Econometric Models Based on Count Data. Journal of Applied Econometrics, 1(1).

Chen, T., \& Guestrin, C. (2016). XGBoost. Proceedings of KDD 2016.

Denuit, M., et al. (2007). Actuarial Modelling of Claim Counts. Wiley.

Frees, E. W., \& Valdez, E. A. (2008). Hierarchical Insurance Claims Modeling. JASA, 103(484).

Klugman, S. A., et al. (2012). Loss Models: From Data to Decisions. Wiley.

McCullagh, P., \& Nelder, J. A. (1989). Generalised Linear Models. Chapman and Hall.

Noll, A., et al. (2018). Case Study: French Motor Third-Party Liability Claims. SSRN.

Wüthrich, M. V. (2019). Insurance Analytics with Actuarial Applications. Swiss Association of Actuaries.



License

This project is licensed under the MIT License.

Developed as part of a Research Master's scholarship portfolio in Actuarial Data Science.

