# Script for comparing synthetic and original data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def aggregate_one_hot_columns(df, prefix):
    """
    Combine one-hot encoded columns back into a single categorical column.

    Args:
        df (pd.DataFrame): DataFrame containing one-hot encoded columns.
        prefix (str): Prefix of the original column (e.g., 'gender', 'residence').

    Returns:
        pd.Series: Combined categorical column.
    """
    cols = [col for col in df.columns if col.startswith(prefix + "_")]
    if not cols:
        return None

    return df[cols].idxmax(axis=1).str[len(prefix) + 1 :]


def compare_numerical_features(real_data, synthetic_data, numerical_cols):
    """
    Compare numerical features using distribution plots.

    Args:
        real_data (pd.DataFrame): Original dataset.
        synthetic_data (pd.DataFrame): Synthetic dataset.
        numerical_cols (list): List of numerical columns to compare.

    Returns:
        None
    """
    for col in numerical_cols:
        if col not in real_data.columns or col not in synthetic_data.columns:
            print(
                f"[WARNING] Numerical column '{col}' is missing in one of the datasets. Skipping..."
            )
            continue
        plt.figure(figsize=(8, 6))
        sns.kdeplot(real_data[col], label="Original", shade=True, color="blue")
        sns.kdeplot(synthetic_data[col], label="Synthetic", shade=True, color="orange")
        plt.title(f"Distribution Comparison: {col}")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend()
        plt.show()


def compare_categorical_features(real_data, synthetic_data, categorical_cols):
    """
    Compare categorical features using bar plots.

    Args:
        real_data (pd.DataFrame): Original dataset.
        synthetic_data (pd.DataFrame): Synthetic dataset.
        categorical_cols (list): List of categorical columns to compare.

    Returns:
        None
    """
    for col in categorical_cols:
        real_col = aggregate_one_hot_columns(real_data, col)
        synthetic_col = aggregate_one_hot_columns(synthetic_data, col)

        if real_col is None or synthetic_col is None:
            print(
                f"[WARNING] Categorical column '{col}' is missing or not encoded properly. Skipping..."
            )
            continue

        plt.figure(figsize=(8, 6))
        real_counts = real_col.value_counts(normalize=True).sort_index()
        synthetic_counts = synthetic_col.value_counts(normalize=True).sort_index()

        df_comparison = pd.DataFrame(
            {"Original": real_counts, "Synthetic": synthetic_counts}
        ).fillna(0)

        df_comparison.plot(kind="bar", figsize=(10, 6))
        plt.title(f"Frequency Comparison: {col}")
        plt.xlabel(col)
        plt.ylabel("Normalized Frequency")
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # Paths to datasets
    real_data_path = r"C:\SCHOOL\FALL 24\STATS 161\final\final\data\processed_data.csv"
    synthetic_data_path = (
        r"C:\SCHOOL\FALL 24\STATS 161\final\final\data\synthetic_data.csv"
    )

    # Load datasets
    print("Loading datasets...")
    real_data = pd.read_csv(real_data_path)
    synthetic_data = pd.read_csv(synthetic_data_path)

    # Define numerical and categorical columns
    numerical_cols = [
        "age",
        "device_size",
        "app_score",
        "u_refreshTimes",
        "u_feedLifeCycle",
    ]
    categorical_cols = ["gender", "residence", "city", "device_name", "net_type"]

    # Compare numerical features
    print("\nComparing numerical features...")
    compare_numerical_features(real_data, synthetic_data, numerical_cols)

    # Compare categorical features
    print("\nComparing categorical features...")
    compare_categorical_features(real_data, synthetic_data, categorical_cols)
