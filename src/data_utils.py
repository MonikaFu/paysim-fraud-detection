def get_temporal_train_test_sets(df, date_column="step", test_size=0.2):
    """
    Splits the dataset into training and testing sets preserving the temporal order.

    Parameters:
        df (pd.DataFrame): The input DataFrame to split.
        date_column (str): The name of the column containing the temporal information.
        test_size (float): The proportion of the dataset to include in the test split.
    Returns:
        tuple: A tuple containing the training and testing sets (train_df, test_df).
    """

    # Sort the DataFrame by the specified date column to preserve temporal order
    df_sorted = df.sort_values(by=date_column)

    # Calculate the index for splitting
    split_index = int(len(df_sorted) * (1 - test_size))

    # Split the DataFrame into training and testing sets
    train_df = df_sorted.iloc[:split_index]
    test_df = df_sorted.iloc[split_index:]

    known_defrauded_customers = train_df[train_df['isFraud'] == 1]['nameOrig'].unique()

    test_df = test_df[~test_df['nameOrig'].isin(known_defrauded_customers)]

    return train_df, test_df