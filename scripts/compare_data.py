import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# WE CHANGE THESE PATH VALUES HERE
INPUT_PATH = "data/raw_train_data_ads_100k.csv"
OUTPUT_PATH = "data/synthetic_fair_data_ads_100k.csv"


def compare_numerical_features(real_data, synthetic_data, numerical_cols):
    for col in numerical_cols:
        if col not in real_data.columns or col not in synthetic_data.columns:
            print(
                f"[UH OH SPAGETTIO] numerical column '{col}' is missing. We skipping..."
            )
            continue

        # plotting distribution configs
        plt.figure(figsize=(8, 7))
        sns.kdeplot(real_data[col], label="Original", shade=True, color="blue")
        sns.kdeplot(synthetic_data[col], label="Synthetic", shade=True, color="orange")
        plt.title(f"Distribution Comparison: {col}")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend()
        plt.show()


def compare_categorical_features(real_data, synthetic_data, categorical_cols):

    for col in categorical_cols:
        if col not in real_data.columns or col not in synthetic_data.columns:
            print(
                f"[UH OH SPAGETTIO] categorical column '{col}' is missing. We skipping..."
            )
            continue

        # plotting distribution configs
        plt.figure(figsize=(8, 7))
        real_counts = real_data[col].value_counts(normalize=True).sort_index()
        synthetic_counts = synthetic_data[col].value_counts(normalize=True).sort_index()

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

    # we set our declared paths here
    real_data_path = INPUT_PATH
    synthetic_data_path = OUTPUT_PATH

    # we load the datasets
    print("loading our datasets...")
    real_data = pd.read_csv(real_data_path)
    synthetic_data = pd.read_csv(synthetic_data_path)

    # we need to convert to strings so that TabFairGAN can process them other wise it breaks
    categorical_columns = [
        "gender",
        "residence",
        "city",
        "city_rank",
        "series_dev",
        "series_group",
        "emui_dev",
        "device_name",
        "net_type",
        "label",
    ]
    for col in categorical_columns:
        real_data[col] = real_data[col].astype(str)
        synthetic_data[col] = synthetic_data[col].astype(str)

    # we define the numerical columns so we can compare them
    numerical_cols = [
        "age",
        "device_size",
        "app_score",
        "u_refreshTimes",
        "u_feedLifeCycle",
    ]

    # compare numerical features
    print("\nComparing numerical features...")
    compare_numerical_features(real_data, synthetic_data, numerical_cols)

    # compare categorical features
    print("\nComparing categorical features...")
    compare_categorical_features(real_data, synthetic_data, categorical_columns)
