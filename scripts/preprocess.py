import pandas as pd
from utils import encode_categorical, normalize_numerical

# Originally, this was used with another tab gan library
categorical_cols = [
    "gender",
    "residence",
    "city",
    "city_rank",
    # "series_dev",
    # "series_group",
    # "emui_dev",
    # "device_name",
    # "net_type",
]

numerical_cols = [
    "age",
    # "device_size",
    # "app_score",
    "u_refreshTimes",
    "u_feedLifeCycle",
]

ignored_fields = [
    "log_id",
    "user_id",
    "pt_d",
    "task_id",
    "adv_id",
    "creat_type_cd",
    "adv_prim_id",
    "inter_type_cd",
    "slot_id",
    "site_id",
    "spread_app_id",
    "hispace_app_tags",
    "app_second_class",
    "ad_click_list_v001",
    "ad_click_list_v002",
    "ad_click_list_v003",
    "ad_close_list_v001",
    "ad_close_list_v002",
    "ad_close_list_v003",
    "u_newsCatInterestsST",
]

include_fields = categorical_cols + numerical_cols


def preprocess_data(input_path, output_path):
    # Load the dataset
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)

    # Drop ignored fields
    df = df.drop(columns=ignored_fields, errors="ignore")

    # Validate fields
    missing_categorical = [col for col in categorical_cols if col not in df.columns]
    missing_numerical = [col for col in numerical_cols if col not in df.columns]
    if missing_categorical or missing_numerical:
        raise ValueError(
            f"Missing fields in the dataset:\n"
            f"Categorical: {missing_categorical}\nNumerical: {missing_numerical}"
        )

    # Preprocess categorical fields
    print("Encoding categorical fields...")
    df, encoders = encode_categorical(df, categorical_cols)

    # Preprocess numerical fields
    print("Normalizing numerical fields...")
    df, scaler = normalize_numerical(df, numerical_cols)

    # Save the processed dataset
    print(f"Saving processed dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Processed data saved successfully at {output_path}")


if __name__ == "__main__":
    # Define input/output paths
    input_file = "data/raw_data.csv"
    output_file = "data/processed_data.csv"

    preprocess_data(input_file, output_file)
