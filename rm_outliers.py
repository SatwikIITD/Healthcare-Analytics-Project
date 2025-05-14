import numpy as np
import pandas as pd

def filter_dataset(x_train, y_train, alpha=0.5):
    """
    Combine x_train and y_train, calculate p = total_costs / length_of_stay, 
    and filter rows based on the range (mean(p) ± 0.5 * std_dev(p)).
    
    Args:
        x_train (pd.DataFrame): Features dataset, must include 'length_of_stay'.
        y_train (pd.Series or np.ndarray): Target variable `total_costs`.
        
    Returns:
        pd.DataFrame: Filtered dataset.
    """
    # Combine x_train and y_train into a single DataFrame
    dataset = x_train.copy()
    dataset['total_costs'] = y_train

    # Ensure 'length_of_stay' exists in x_train
    if 'length_of_stay' not in dataset.columns:
        raise ValueError("The 'length_of_stay' column is required in x_train.")

    # Calculate p = total_costs / length_of_stay
    dataset['p'] = dataset['total_costs'] / dataset['length_of_stay']

    # Handle potential division by zero
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna(subset=['p'])

    # Compute mean and standard deviation of p
    p_mean = dataset['p'].mean()
    p_std = dataset['p'].std()

    # Define the filtering range
    lower_bound = p_mean - alpha * p_std
    upper_bound = p_mean + alpha * p_std

    # Filter the dataset based on the range
    filtered_dataset = dataset[(dataset['p'] >= lower_bound) & (dataset['p'] <= upper_bound)]

    # Drop the 'p' column (optional, depending on whether you need it)
    filtered_dataset = filtered_dataset.drop(columns=['p'])
    filtered_dataset.to_csv(f'filtered_{alpha}.csv',index=False)

    initial_count = len(dataset)
    # Calculate the final number of rows
    final_count = len(filtered_dataset)

    # Calculate and print the percentage of rows removed
    rows_removed = initial_count - final_count
    percentage_removed = (rows_removed / initial_count) * 100
    print(f"Percentage of rows removed: {percentage_removed:.2f}%")

# Example usage
if __name__ == "__main__":
    # Example data
    x_train = pd.read_csv('x_train.csv')
    y_train = pd.read_csv('y_train.csv').to_numpy().flatten()

    # Filter the dataset
    filter_dataset(x_train, y_train,alpha = 0.5)

    # Print the results
    print("Filtered")
