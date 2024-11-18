# Utility functions for encoding, scaling, etc.

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


# Encode explicitly defined categorical fields
def encode_categorical(df, categorical_cols):
    """
    Encode categorical fields using OneHotEncoder.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        categorical_cols (list): List of column names to encode.

    Returns:
        pd.DataFrame: DataFrame with encoded categorical columns.
        dict: Dictionary of encoders for each column.
    """
    encoders = {}
    for col in categorical_cols:
        if col not in df.columns:
            raise ValueError(f"Categorical column '{col}' not found in the DataFrame.")

        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        transformed = encoder.fit_transform(df[[col]])
        encoded_df = pd.DataFrame(
            transformed, columns=[f"{col}_{cat}" for cat in encoder.categories_[0]]
        )
        df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1).drop(
            columns=[col]
        )
        encoders[col] = encoder
    return df, encoders


# Normalize explicitly defined numerical fields
def normalize_numerical(df, numerical_cols):
    """
    Normalize numerical fields using MinMaxScaler.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        numerical_cols (list): List of column names to normalize.

    Returns:
        pd.DataFrame: DataFrame with normalized numerical columns.
        MinMaxScaler: Scaler object used for normalization.
    """
    scaler = MinMaxScaler()
    for col in numerical_cols:
        if col not in df.columns:
            raise ValueError(f"Numerical column '{col}' not found in the DataFrame.")

    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler


# Prepare dataset for TabGAN
def prepare_dataset(df, n_sample):
    """
    Prepare the dataset for TabGAN by splitting it into train, target, and test sets.

    Args:
        df (pd.DataFrame): Processed DataFrame.
        n_sample (int): Number of rows for training data.

    Returns:
        pd.DataFrame: Training data.
        pd.DataFrame: Target labels (simulated).
        pd.DataFrame: Testing data.
    """
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=12939).reset_index(drop=True)

    # Split the data
    train = df.iloc[:n_sample]  # First `n_sample` rows as training
    test = df.iloc[n_sample : n_sample + 20]  # Next 20 rows as testing

    # Simulated binary labels (50% `0`s and `1`s)
    target = pd.DataFrame([0] * (n_sample // 2) + [1] * (n_sample // 2)).sample(
        frac=1, random_state=42
    )

    return train, target, test
