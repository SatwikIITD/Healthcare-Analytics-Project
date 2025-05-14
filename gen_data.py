import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os

numerical_columns = ["length_of_stay", "total_costs"]

categorical_columns = [
    "hospital_service_area",
    "hospital_county",
    "operating_certificate_number",
    "facility_name",
    "age_group",
    "gender",
    "race",
    "ethnicity",
    "type_of_admission",
    "patient_disposition",
    "ccsr_diagnosis_code",
    "ccsr_procedure_code",
    "ccs_diagnosis_code",
    "ccs_procedure_code",
    "apr_drg_code",
    "apr_mdc_code",
    "apr_severity_of_illness",
    "apr_risk_of_mortality",
    "apr_medical_surgical",
    "payment_typology_1",
    "emergency_department_indicator",
]

def load_and_clean_data(i):
    # Read the dataset
    # df = pd.read_csv('/kaggle/input/nih-spark/nih/full_dataset.csv')
    df = pd.read_csv(f"/kaggle/input/nih-rm-ds/nih_rm/sparcs_{i}.csv")

    # Replace '120 +' with '120' in length_of_stay and convert to numeric
    df["length_of_stay"] = df["length_of_stay"].replace("120 +", "120")
    df["length_of_stay"] = pd.to_numeric(df["length_of_stay"])

    if i == 2022:
        df["total_costs"] = df["total_costs"].str.replace(",", "").astype(float)
    
    # present_categorical = list(set(categorical_columns) & set(df.columns))
    present_numerical = [col for col in numerical_columns if col in df.columns]
    present_categorical = [col for col in categorical_columns if col in df.columns]

    # Keep only specified columns
    df = df[present_numerical + present_categorical]

    return df


# Modified scaling section
def selective_scaling(X_train, X_test, target_encode_features):
    """
    Selectively scale only target encoded features and length_of_stay
    """
    # Initialize scaler
    scaler = StandardScaler()

    # Features to scale (target encoded features + length_of_stay)
    features_to_scale = target_encode_features + ["length_of_stay"]

    # Create copies to avoid modifying original data
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # Scale only selected features
    X_train_scaled[features_to_scale] = scaler.fit_transform(
        X_train[features_to_scale]
    ).astype(np.float32)
    X_test_scaled[features_to_scale] = scaler.transform(
        X_test[features_to_scale]
    ).astype(np.float32)

    return X_train_scaled, X_test_scaled

def calculate_target_ratios(df, categorical_cols):
    """Calculate mean_total_costs/mean_length_of_stay for each category"""
    encoding_dict = {}

    df["length_of_stay"].fillna(df["length_of_stay"].mean(), inplace=True)

    for col in categorical_cols:
        # Group by category and calculate means
        group_stats = df.groupby(col).agg(
            {"total_costs": "mean", "length_of_stay": "mean"}
        )
        # Calculate ratio
        encoding_dict[col] = (
            group_stats["total_costs"] / group_stats["length_of_stay"]
        ).to_dict()
    return encoding_dict

def encode_and_transform(df, encoding_dict, categorical_cols):
    """Apply encoding and multiply by length_of_stay"""
    df_encoded = df.copy()
    # df_encoded["length_of_stay"].fillna(
    #     df_encoded["length_of_stay"].mean(), inplace=True
    # )

    mean_total_costs = df_encoded["total_costs"].mean()

    # Replace categories with their target ratios
    for col in categorical_cols:

        if np.nan in encoding_dict[col]:
            encoding_dict[col][np.nan] = mean_total_costs

        df_encoded[col] = df_encoded[col].map(encoding_dict[col])
        # Handle unseen categories with mean of known categories
        if df_encoded[col].isnull().any():
            mean_ratio = np.mean(list(encoding_dict[col].values()))
            df_encoded[col].fillna(mean_ratio, inplace=True)

        # Multiply by length_of_stay
        df_encoded[col] = df_encoded[col] * df_encoded["length_of_stay"]

    return df_encoded

def get_X_y_train(i,rand_seed):
    # Load and clean data
    print("Loading and cleaning data...")
    df = load_and_clean_data(i)

    # Split into train and test
    X = df.drop("total_costs", axis=1)
    y = df["total_costs"]

    X_train, _, y_train,_ = train_test_split(
        X, y, test_size=0.2, random_state=rand_seed
    )

    return X_train,y_train

def get_X_y_test(i,rand_seed):
    # Load and clean data
    print("Loading and cleaning data...")
    df = load_and_clean_data(i)

    # Split into train and test
    X = df.drop("total_costs", axis=1)
    y = df["total_costs"]

    _, X_test, _,y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_seed
    )

    return X_test,y_test

os.makedirs("test_train_split", exist_ok=True)
# Run the pipeline
print("Starting preprocessing pipeline...")
for i in range(2014, 2022):
    for rand_seed in range(0,1):
        # X_train,y_train= get_X_y_train(i,rand_seed)
        X_test,y_test= get_X_y_test(i,rand_seed)
        
        os.makedirs(f"test_train_split/sparks_{i}/rand_seed_{rand_seed}", exist_ok=True)
        # Save the processed datasets
        print(f"\nSaving processed datasets,i={i}...,rand_seed={rand_seed}")

        # X_train.to_csv(f"test_train_split/sparks_{i}/rand_seed_{rand_seed}/x_train.csv", index=False)
        X_test.to_csv(f"test_train_split/sparks_{i}/rand_seed_{rand_seed}/x_test.csv", index=False)
        # y_train.to_csv(f"test_train_split/sparks_{i}/rand_seed_{rand_seed}/y_train.csv", index=False)
        y_test.to_csv(f"test_train_split/sparks_{i}/rand_seed_{rand_seed}/y_test.csv", index=False)
        
        print("Datasets saved successfully!")

for i in range(2014, 2015):
    for rand_seed in range(0,1):
        X_train,y_train= get_X_y_train(i,rand_seed)
        # X_test,y_test= get_X_y_test(i,rand_seed)
        
        os.makedirs(f"test_train_split/sparks_{i}/rand_seed_{rand_seed}", exist_ok=True)
        # Save the processed datasets
        print(f"\nSaving processed datasets,i={i}...,rand_seed={rand_seed}")

        X_train.to_csv(f"test_train_split/sparks_{i}/rand_seed_{rand_seed}/x_train.csv", index=False)
        # X_test.to_csv(f"test_train_split/sparks_{i}/rand_seed_{rand_seed}/x_test.csv", index=False)
        y_train.to_csv(f"test_train_split/sparks_{i}/rand_seed_{rand_seed}/y_train.csv", index=False)
        # y_test.to_csv(f"test_train_split/sparks_{i}/rand_seed_{rand_seed}/y_test.csv", index=False)
        
        print("Datasets saved successfully!")
   
