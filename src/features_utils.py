import pandas as pd
import numpy as np

def add_log_amount(df: pd.DataFrame, column_name: str = 'amount'):
    """
    Adds a new column 'log_amount' to the DataFrame, which is the natural logarithm of the specified column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the specified column.
    column_name (str): The name of the column to apply the logarithm to.
    
    Returns:
    pd.DataFrame: The DataFrame with the added 'log_amount' column.
    """
    df['log_amount'] = np.log1p(df[column_name])  # Adding 1 to avoid log(0)
    return df

def add_transation_type_dummies(df: pd.DataFrame, column_name: str = 'type'):
    """
    Adds dummy variables for the specified categorical column to the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the specified column.
    column_name (str): The name of the categorical column to create dummies for.
    
    Returns:
    pd.DataFrame: The DataFrame with the added dummy variables.
    """
    dummies = pd.get_dummies(df[column_name], prefix=column_name, dtype=float)
    df = pd.concat([df, dummies], axis=1)
    return df