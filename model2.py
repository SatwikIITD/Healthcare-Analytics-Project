import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os 
import time 
inflation_factors = []
os.makedirs('low_conf_indices',exist_ok=True)
for year in range(2013,2022):
    start = time.time()
    # Step 1: Load Preprocessed Train and Test Datasets
    X_train = pd.read_csv(f"test_train_split/sparks_{year}/x_train_target.csv")  # Features for training
    X_test = pd.read_csv(f"test_train_split/sparks_{year}/x_test_target.csv")    # Features for testing
    y_train = pd.read_csv(f"test_train_split/sparks_{year}/y_train.csv").squeeze()  # Target for training
    y_test = pd.read_csv(f"test_train_split/sparks_{year}/y_test.csv").squeeze()    # Target for testing

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # Step 2: Train an Ensemble of CatBoost Models
    n_models = 10  # Number of models in the ensemble
    models = []

    for i in range(n_models):
        model = CatBoostRegressor(
            random_seed=42 + i,
            verbose=False  # Suppress training output
        )
        model.fit(X_train, y_train)
        models.append(model)

    # Step 3: Make Predictions with the Ensemble
    predictions = np.array([model.predict(X_test) for model in models])
    # Calculate mean prediction and uncertainty (standard deviation)
    pred_mean = predictions.mean(axis=0)
    pred_std = predictions.std(axis=0)

    # Step 4: High-Confidence Filtering Based on Uncertainty
    confidence_threshold = 3000  # Adjust this threshold based on your requirements
    # Define high-confidence predictions
    high_confidence_mask = pred_std <= confidence_threshold
    # Note indices of low-confidence predictions
    low_confidence_indices = np.where(~high_confidence_mask)[0]  # Indices of rows not predicted

    # Debug: Check if the mask has valid samples
    if high_confidence_mask.sum() == 0:
        print("No samples meet the high-confidence criteria. Consider adjusting the threshold.")
    else:
        # Filter high-confidence predictions
        X_test_high_confidence = X_test[high_confidence_mask]
        y_test_high_confidence = y_test[high_confidence_mask]
        # X_test_high_confidence.to_csv("X_test_hc.csv", index=False)
        # y_test_high_confidence.to_csv("y_test_hc.csv", index=False)
        # print("Test data saved successfully!")

        # # Save high-confidence examples to a new CSV file
        # high_confidence_data = X_test_high_confidence.copy()
        # high_confidence_data['total_costs'] = y_test_high_confidence
        # high_confidence_data.to_csv("high_confidence_predictions.csv", index=False)
        # print(f"Filtered {len(high_confidence_data)} high-confidence examples saved to 'high_confidence_predictions.csv'.")

        # Save low-confidence indices to a pickle file
        with open(f"low_conf_indices/sparks_{year}.pkl", "wb") as f:
            pickle.dump(low_confidence_indices.tolist(), f)

        # Evaluate Performance (Only for high-confidence predictions)
        mae = mean_absolute_error(y_test_high_confidence, pred_mean[high_confidence_mask])
        mse = mean_squared_error(y_test_high_confidence, pred_mean[high_confidence_mask])
        rmse = np.sqrt(mse)
        coverage = high_confidence_mask.mean()  # Proportion of predictions made

        print(f"Mean Absolute Error (High Confidence): {mae}")
        print(f"Root Mean Squared Error (High Confidence): {rmse}")
        print(f"Coverage (Proportion of Predictions Made): {coverage}")
        end = time.time()
        print(f"time_taken={end-start}s")
