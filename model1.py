import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import torch
import time
import matplotlib.pyplot as plt
from xgboost.callback import TrainingCallback
import pickle
from collections import defaultdict

results = defaultdict(dict)

class ProgressCallback(TrainingCallback):
    def __init__(self, model_name):
        self.model_name = model_name
        self.start_time = None
        self.epoch_times = []
        self.train_rmse = []
        self.test_rmse = []

    def after_iteration(self, model, epoch, evals_log):
        if not self.start_time:
            self.start_time = time.time()
        
        epoch_time = time.time() - self.start_time
        self.epoch_times.append(epoch_time)
        
        train_rmse = evals_log.get('train', {}).get('rmse')
        test_rmse = evals_log.get('test', {}).get('rmse')
        
        if train_rmse:
            self.train_rmse.append(train_rmse[-1])
        if test_rmse:
            self.test_rmse.append(test_rmse[-1])
        
        if (epoch + 1) % 10 == 0:
            print(f"{self.model_name} - Iteration {epoch + 1}: "
                  f"Training RMSE: {train_rmse[-1]:.4f}, Test RMSE: {test_rmse[-1]:.4f}")
        return False

def check_gpu():
    """Check GPU availability and print device info"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU available: {gpu_name}")
        return True
    else:
        print("No GPU found, falling back to CPU")
        return False

def prepare_data(df):
    """Prepare the numerical dataset for modeling"""
    X = df.drop('total_costs', axis=1)
    y = df['total_costs']
    return X, y

def calculate_trimmed_rmse(y_true, y_pred, percentile=0.95, log_file='errors_log_sorted.txt'):
    """Calculate RMSE on best predictions and log sorted errors to a file."""
    # Calculate absolute errors
    errors = np.abs(y_pred - y_true)
    
    # Sort the errors in descending order for logging
    sorted_errors = np.sort(errors)[::-1]
    
    # # Write all errors to a text file
    # with open(log_file, 'w') as f:
    #     f.write("Sorted Errors (Descending Order):\n")
    #     for i, error in enumerate(sorted_errors):
    #         f.write(f"Rank {i + 1}: Error: {error}\n")
    
    print(f"All sorted errors have been logged to {log_file}.")
    
    # Calculate trimmed RMSE
    n = len(sorted_errors)
    n_keep = int(n * percentile)
    trimmed_mse = np.mean(sorted_errors[-n_keep:] ** 2)
    return np.sqrt(trimmed_mse)

def train_and_evaluate_models(X_train, X_test, y_train, y_test, year,rand_seed,use_gpu=False):
    """Train and evaluate models with GPU support"""
    # Convert to DMatrix format
    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, y_test)
    
    # XGBoost parameters
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 10,
        'learning_rate': 0.05,
        'tree_method': 'gpu_hist' if use_gpu else 'hist',
        'predictor': 'gpu_predictor' if use_gpu else 'cpu_predictor',
        'gpu_id': 0 if use_gpu else None,
        'verbosity': 0,  # Silences all logs except errors
    }
    
    # Train XGBoost
    print("\nTraining XGBoost model...")
    start_time = time.time()
    
    # Create and train the model
    model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=3000,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        verbose_eval=False,  # Suppresses evaluation logs
    )
    
    training_time = time.time() - start_time
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(dtest)
    
    # Calculate metrics
    if isinstance(y_test, pd.DataFrame) or isinstance(y_test, pd.Series):
        y_test = y_test.to_numpy().flatten()
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    trimmed_rmse = calculate_trimmed_rmse(y_test, y_pred)
    
    results[f"{year}"][f"{rand_seed}"] = {
        'MAE': round(mae,2),
        'RMSE': round(rmse,2),
        'Trimmed RMSE': round(trimmed_rmse,2),
        'Training Time (seconds)': round(training_time,2)
    }

    print(f"XGBoost training completed in {training_time:.2f} seconds")
            
    return model

# Modified scaling section
def selective_scaling_train(X_train,target_encode_features):
    """
    Selectively scale only target encoded features and length_of_stay
    """
    # Initialize scaler
    scaler = StandardScaler()

    # Features to scale (target encoded features + length_of_stay)
    features_to_scale = target_encode_features + ["length_of_stay"]

    # Create copies to avoid modifying original data
    X_train_scaled = X_train.copy()
    
    # Scale only selected features
    X_train_scaled[features_to_scale] = scaler.fit_transform(
        X_train[features_to_scale]
    ).astype(np.float32)
    
    return X_train_scaled, scaler

def selective_scaling_test(X_test,target_encode_features, scaler):
    # Features to scale (target encoded features + length_of_stay)
    features_to_scale = target_encode_features + ["length_of_stay"]

    # Create copies to avoid modifying original data
    X_test_scaled = X_test.copy()

    X_test_scaled[features_to_scale] = scaler.transform(
        X_test[features_to_scale]
    ).astype(np.float32)

    return X_test_scaled

def main(year, final_year, rand_seed):
    # Add this line to access the global variable
    global results
    
    # Check GPU availability
    use_gpu = check_gpu()
    
    X_train = pd.read_csv(f'/kaggle/working/test_train_split/sparks_{year}/rand_seed_{rand_seed}/x_train.csv')
    y_train = pd.read_csv(f'/kaggle/working/test_train_split/sparks_{year}/rand_seed_{rand_seed}/y_train.csv')
    target_encode_columns = [col for col in categorical_columns if col in X_train.columns]
    print(target_encode_columns)
    final_features = target_encode_columns + ['length_of_stay']
    print(final_features)
    
    train_df = pd.concat([X_train, y_train], axis=1)
    
    encoding_dict = calculate_target_ratios(train_df, target_encode_columns)
    train_df = encode_and_transform(train_df, encoding_dict, target_encode_columns)
    X_train = train_df.drop("total_costs", axis=1)
    y_train = train_df["total_costs"]
    
    for test_year in range(year, final_year+1):
        X_test = pd.read_csv(f"test_train_split/sparks_{test_year}/rand_seed_{rand_seed}/x_test.csv")
        y_test = pd.read_csv(f"test_train_split/sparks_{test_year}/rand_seed_{rand_seed}/y_test.csv")
        X_test = X_test[X_train.columns]
    
        test_df = pd.concat([X_test, y_test], axis=1)
            
        print("Applying target encoding...")
        test_df = encode_and_transform(test_df, encoding_dict, target_encode_columns)
    
        X_test = test_df.drop("total_costs", axis=1)
        y_test = test_df["total_costs"]
    
        print("Scaling target encoded features and length_of_stay...")
        X_train, scaler = selective_scaling_train(X_train, target_encode_columns)
        X_test = selective_scaling_test(X_test, target_encode_columns, scaler)
        
        # Train and evaluate models
        # This function returns the model, not metrics
        model = train_and_evaluate_models(X_train, X_test, y_train, y_test, test_year, rand_seed, use_gpu=use_gpu)
        
        # The metrics are already stored in the results dictionary by train_and_evaluate_models
        # No need to access model_results as a dictionary
        print(results[str(test_year)])
        
# years = np.arange(2014, 2022)  # Years 2013 to 2021
# for year in years:
train_year=2014
final_year=2021
for rand_seed in range(0,1):
    main(train_year,final_year,rand_seed)

mae_lst = []
trimmed_rmse_lst = []

for year, seeds in results.items():
    mae_values = []
    trimmed_rmse_values = []

    for rand_seed, metrics in seeds.items():
        mae_values.append(round(metrics["MAE"],2))
        trimmed_rmse_values.append(round(metrics["Trimmed RMSE"],2))

    # Compute mean and append to lists
    if mae_values:
        mae_lst.append(np.mean(mae_values))
    if trimmed_rmse_values:
        trimmed_rmse_lst.append(np.mean(trimmed_rmse_values))
    
print("mae_lst=",mae_lst)
print("trimmed_rmse_lst=",trimmed_rmse_lst)

years = [];
for i in range(train_year,final_year+1):
    years.append(i)
# Plot Trimmed RMSE comparison
plt.figure(figsize=(10, 5))
# plt.plot(years, model1_trimmed_rmse_same[:(len(years))], marker="o", linestyle="-", label="Trained on same year's data", color="b")
plt.plot(years, trimmed_rmse_lst, marker="s", linestyle="--", label=f"Trained on {year} data", color="r")
plt.title("Model 1 - Trimmed RMSE")
plt.xlabel("Year")
plt.ylabel("Trimmed RMSE")
plt.xticks(years)
plt.legend()
plt.grid(True)
plt.savefig('model1_trimmed_rmse.png')
plt.show()

# Plot MAE comparison
plt.figure(figsize=(10, 5))
# plt.plot(years, model1_mae_same[:(len(years))], marker="o", linestyle="-", label="Trained on same year's data", color="b")
plt.plot(years, mae_lst, marker="s", linestyle="--", label=f"Trained on {year} data", color="r")
plt.title("Model 1 - MAE")
plt.xlabel("Year")
plt.ylabel("MAE")
plt.xticks(years)
plt.legend()
plt.grid(True)
plt.savefig('model1_mae.png')
plt.show()
