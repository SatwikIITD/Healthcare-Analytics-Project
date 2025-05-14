# Healthcare-Analytics-Project

This repository contains the work done by **Satwik** and **Aneeket Yadav** under the supervision of **Prof. Ravi Rao** and **Prof. Rahul Garg** from **January to April 2025** on the **Application of Machine Learning in Healthcare Analytics**.

##  Research Paper

**Title:** *Utilizing Machine Learning to Improve Healthcare Cost Prediction on Large Public Datasets*  
**Conference:** Accepted at the **2025 IEEE 5th International Conference on Smart Information Systems and Technologies (SIST)**

##  Project Overview

While traditional approaches to healthcare cost prediction have relied on ensemble methods like random forests and gradient boosting, our focus is on building **high-confidence predictive models** that are not only accurate but also **aware of their uncertainty**.

To this end, we propose two complementary models:

### Model 1: Error-Minimized Prediction

- Trained to minimize **Root Mean Squared Error (RMSE)**.
- Post-prediction, the **top 1% of outliers with highest squared error** are discarded to report a more robust RMSE.
- This serves as a baseline for evaluating cost prediction performance.

### Model 2: Confidence-Aware Ensemble Predictor

- Utilizes an **ensemble of CatBoost regressors**.
- For each input, the model decides **whether to predict or abstain** based on internal confidence measures.
- Introduces a **coverage-error tradeoff**:
  - **Higher coverage** may increase prediction error.
  - **Lower coverage** allows the model to make **more confident predictions**.
- For practical utility, we tune hyperparameters to ensure the model predicts for **>98% of the inputs**.
  - This threshold can be adjusted based on the **risk tolerance** and **requirements** of the healthcare provider or hospital.

The goal of Model 2 is to **approximate the performance** of Model 1 while providing **confidence-aware outputs**, making it suitable for real-world deployment in critical settings where **low-confidence predictions can be costly**.

## Repository Contents

- `model1.py`: Code for training the catboost model 
- `model2.py`: Code for training the ensemble predictor and implementing the confidence thresholding mechanism.
- `fetch_data.py`: Script to fetch the dataset via API call.
- `results/`: Evaluation metrics, plots, and comparisons between models.
- `paper/`: Drafts and final version of the accepted research paper.



