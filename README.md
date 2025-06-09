# Credit-Card-Fraud-Detection-and-Prevention-Hackathon

This repository contains the complete analysis and machine learning models developed for the **Credit Card Fraud Detection and Prevention Hackathon**. The project focuses on identifying fraudulent credit card transactions using a highly imbalanced dataset and proposes a strategy for real-time detection.

![Hackathon Banner](https://github.com/devtlv/studentsGitHub/blob/master/Images%20cours%20%C3%A0%20h%C3%A9berger%20-%2006.03.2024/Week%204%20-%20Databases/W4D5/banner-hackathon-design-sprintlike-event-260nw-1418226719.jpg?raw=true)

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Findings & Visualizations](#key-findings--visualizations)
- [Model Performance](#model-performance)
- [Feature Importance](#feature-importance)
- [Tools & Libraries Used](#tools--libraries-used)
- [How to Run This Project](#how-to-run-this-project)
- [Fraud Prevention Strategy](#fraud-prevention-strategy)
- [Future Work](#future-work)

## Project Overview

The objective of this project is to analyze a dataset of credit card transactions and build a machine learning model capable of accurately detecting fraudulent activities. Given the nature of financial data, the primary challenges were handling anonymized features and a severe class imbalance, where fraudulent transactions constitute a very small minority of the data.

Our solution moves beyond simple accuracy metrics to focus on **precision** and **recall**—critical for minimizing financial loss while ensuring a seamless experience for legitimate customers.

## Problem Statement

Financial institutions face a dual challenge in fraud detection:
1.  **Detecting Fraud:** Accurately identify and block fraudulent transactions to prevent financial losses.
2.  **Minimizing False Positives:** Avoid incorrectly flagging legitimate transactions, which leads to poor customer experience and high operational costs for manual review.

This project addresses this by developing a model optimized to achieve high recall (catch most fraud) while maintaining exceptional precision (avoid false alarms).

## Dataset

The dataset used is the "Credit Card Fraud Detection" dataset from Kaggle, which contains transactions made by European cardholders.

- **Size**: Contains 284,807 transactions.
- **Features**:
    - `Time`: Seconds elapsed between each transaction and the first transaction in the dataset.
    - `Amount`: The transaction amount.
    - `V1-V28`: Anonymized features resulting from a Principal Component Analysis (PCA) to protect user privacy.
- **Target Variable**:
    - `Class`: 1 for fraudulent transactions, 0 otherwise.
- **Key Challenge**: The dataset is highly imbalanced. Fraudulent transactions account for only **0.172%** of all transactions.

## Methodology

Our approach followed a structured data science workflow:

1.  **Exploratory Data Analysis (EDA)**:
    - Analyzed the distribution of `Time`, `Amount`, and the `Class` variable.
    - Visualized the extreme class imbalance to frame the problem.
    - Investigated correlations between features to understand their relationships.

2.  **Data Preprocessing**:
    - **Scaling**: Standardized the `Time` and `Amount` columns using `StandardScaler` to ensure they didn't disproportionately influence the model.
    - **Sub-sampling**: To manage computational resources, a representative sample of 10,000 data points was taken.
    - **Train-Test Split**: The data was split into training (70%) and testing (30%) sets *before* applying any balancing techniques to prevent data leakage.
    - **Handling Class Imbalance**: Applied the **SMOTE (Synthetic Minority Over-sampling Technique)** on the *training data only*. This technique creates synthetic fraudulent samples, providing the model with more examples to learn from without altering the test set's true distribution.

3.  **Predictive Modeling**:
    - Developed two classification models to compare performance:
        1.  **Logistic Regression**: A simple, interpretable baseline model.
        2.  **Random Forest Classifier**: A more complex ensemble model known for its high accuracy and robustness.
    - Evaluated the models using Accuracy, Precision, Recall, F1-Score, and ROC AUC Score.

## Key Findings & Visualizations

### 1. Severe Class Imbalance
The distribution of fraudulent vs. non-fraudulent transactions is stark, highlighting the need for specialized techniques like SMOTE.

*(Insert your Class Distribution chart here)*

### 2. Transaction Amount and Time
EDA revealed that `Amount` and `Time` alone are not sufficient to distinguish fraud. Fraudulent transactions occur across all times of the day and typically involve smaller amounts, but with significant overlap with legitimate transactions.

*(Insert your Time vs. Amount scatter plot here)*

### 3. Hourly Fraud Rate
By converting the `Time` feature to hours, we observed that fraud rates tend to spike during off-peak hours (e.g., early morning), a common pattern as fraudsters exploit lower monitoring periods.

*(Insert your Hourly Fraud Rate line chart here)*

## Model Performance

The Random Forest model significantly outperformed the Logistic Regression baseline, especially in its ability to avoid false positives.

| Metric          | Logistic Regression | Random Forest |
|-----------------|---------------------|---------------|
| **Accuracy**    | 0.99                | **1.00**      |
| **Precision (Fraud)** | 0.40              | **1.00**      |
| **Recall (Fraud)**    | 0.80              | 0.80          |
| **F1-Score (Fraud)**  | 0.53              | 0.89          |
| **ROC AUC Score**   | 0.90              | 0.90          |

![Model Performance Comparison](images/performance_comparison.png) 
*Caption: Random Forest excels with 100% precision, meaning every flagged transaction was genuinely fraudulent.*

The confusion matrices below further illustrate this. The Random Forest model had **zero false positives**, making it highly efficient from an operational standpoint.

*(Insert your confusion matrices for both models side-by-side here)*

## Feature Importance

The Random Forest model allowed us to identify the most influential features in detecting fraud. The top features are `V14`, `V4`, and `V12`. This insight is crucial for building simpler, faster models in the future and for focusing real-time monitoring efforts.

*(Insert your Feature Importance bar chart here)*

## Tools & Libraries Used
- **Python 3**
- **Pandas** & **NumPy**: For data manipulation and numerical operations.
- **Matplotlib** & **Seaborn**: For data visualization.
- **Scikit-learn**: For data preprocessing, modeling (Logistic Regression, Random Forest), and evaluation metrics.
- **Imbalanced-learn**: For implementing SMOTE.
- **Jupyter Notebook**: As the primary development environment.

## How to Run This Project
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/credit-card-fraud-detection.git
    cd credit-card-fraud-detection
    ```
2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You'll need to create a `requirements.txt` file)*

3.  **Download the dataset**:
    - Download `creditcard.csv` from the [Kaggle source](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the root directory.

4.  **Run the Jupyter Notebook**:
    ```bash
    jupyter notebook Credit_Card_Fraud_Detection_and_Prevention_Hackathon.ipynb
    ```

## Fraud Prevention Strategy

Based on our findings, we propose a two-pronged strategy:

1.  **Real-Time Detection**: Deploy the trained Random Forest model as a microservice. New transactions can be passed through the model via an API call. If the model predicts fraud (Class 1), the transaction is flagged for immediate review or automatic blocking, depending on the institution's risk tolerance.

2.  **Enhanced Monitoring & Future-Proofing**:
    - Prioritize monitoring of the top predictive features (`V14`, `V4`, `V12`).
    - Integrate additional data points like IP address location, device ID, and transaction frequency per user to build even more robust features.
    - Regularly retrain the model with new data to adapt to evolving fraud patterns.


