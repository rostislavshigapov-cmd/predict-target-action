# ML Prediction of Target Actions on a Website

## Project Overview

This project is a complete Machine Learning pipeline for predicting whether a website session will result in a target user action.

The model was developed using Google Analytics data containing information about user sessions and events. The main goal was to identify sessions with a high probability of performing valuable actions on the website.

The project covers the full ML workflow — from data preprocessing and feature engineering to model training, evaluation, saving the trained model, and deploying it through a REST API.

## Dataset

The project uses two Google Analytics datasets:

* `ga_sessions.csv` — information about user sessions
* `ga_hits.csv` — information about user events

The dataset contains approximately:

* **1.2M user sessions**
* **4M events**
* **14+ features**

## Target

The target variable indicates whether a user performed one of the defined target actions during a session.

Target actions include:

* `sub_car_claim_click`
* `sub_car_claim_submit_click`
* `sub_open_dialog_click`
* `sub_custom_question_submit_click`
* `sub_call_number_click`
* `sub_callback_submit_click`
* `sub_submit_success`
* `sub_car_request_submit_click`

The task is formulated as a **binary classification problem**.

## Machine Learning

Several approaches were considered during the project, including:

* Logistic Regression
* Random Forest
* CatBoost

The final model is based on **CatBoost**, which was selected due to its ability to efficiently work with categorical features and large tabular datasets.

### Feature Engineering

The preprocessing pipeline includes:

* Date and time feature extraction
* Hour and day of week
* Weekend indicator
* Date components
* Returning user indicator
* Number of hits per session
* Number of unique pages
* Car brand extraction
* Car model extraction
* Processing of categorical traffic, device and geographic features

Categorical features were handled directly by CatBoost.

## Model Evaluation

The model was evaluated using a train/test split and cross-validation.

Final model performance:

| Metric   |    Score |
| -------- | -------: |
| ROC-AUC  | **0.75** |
| Accuracy | **0.70** |

## Deployment

The trained CatBoost model is saved and served through a REST API using **FastAPI**.

The prediction workflow:

```text
User session data
       ↓
Feature preprocessing
       ↓
CatBoost model
       ↓
Prediction
       ↓
FastAPI
       ↓
POST /predict
```

The API accepts session data and returns the model prediction.

## Technologies

* Python
* Pandas
* Scikit-learn
* CatBoost
* FastAPI
* Uvicorn
* Git
* Google Analytics data

## Project Structure

```text
project/
│
├── data/
│   ├── ga_sessions.csv
│   └── ga_hits.csv
│
├── model/
│   └── make_target_action.pkl
│
├── EDA/
│   └── EDA.ipynb
│
├── main.py
├── data_loadind.py
├── features.py
├── train.py
├── config.py
├── convert.json
└── README.md
```

## Key Skills Demonstrated

This project demonstrates practical experience with:

* Exploratory Data Analysis
* Data preprocessing
* Feature engineering
* Binary classification
* Categorical feature handling
* CatBoost
* Model evaluation
* Cross-validation
* Model serialization
* REST API development
* FastAPI
* Git and version control

## Project Goal

The main goal of the project was to build a complete Machine Learning solution from raw data to a working prediction API, demonstrating the ability to apply ML techniques to a real-world tabular dataset.
