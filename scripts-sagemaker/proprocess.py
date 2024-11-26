import pandas as pd
from utils import encode_categorical, normalize_numerical
import boto3
import os

# Explicitly define fields
categorical_cols = [
    "gender",
    "residence",
    "city",
    "city_rank",
    "series_dev",
    "series_group",
    "emui_dev",
    "device_name",
    "net_type",
]

numerical_cols = [
    "age",
    "device_size",
    "app_score",
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


def download_from_s3(s3_path, local_path):
    """
    Download a file from S3 to a local path.

    Args:
        s3_path (str): S3 URI (e.g., "s3://my-bucket/data.csv").
        local_path (str): Local path to save the file.

    Returns:
        None
    """
    s3 = boto3.client("s3")
    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    s3.download_file(bucket, key, local_path)


def upload_to_s3(local_path, s3_path):
    """
    Upload a file from a local path to S3.

    Args:
        local_path (str): Local path to the file.
        s3_path (str): S3 URI (e.g., "s3://my-bucket/data.csv").

    Returns:
        None
    """
    s3 = boto3.client("s3")
    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    s3.upload_file(local_path, bucket, key)


def preprocess_data(input_s3_path, output_s3_path):
    """
    Preprocess the input dataset by selecting specified fields,
    encoding categorical fields, and normalizing numeric fields.

    Args:
        input_s3_path (str): S3 path to the raw dataset CSV.
        output_s3_path (str): S3 path to save the processed dataset CSV.

    Returns:
        None
    """
    # Temporary local paths
    local_input = "/tmp/raw_data.csv"
    local_output = "/tmp/processed_data.csv"

    # Download the dataset from S3
    print(f"Downloading raw dataset from {input_s3_path}...")
    download_from_s3(input_s3_path, local_input)

    # Load the dataset
    print(f"Loading dataset from {local_input}...")
    df = pd.read_csv(local_input)

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

    # Save the processed dataset locally
    print(f"Saving processed dataset locally at {local_output}...")
    df.to_csv(local_output, index=False)

    # Upload the processed dataset to S3
    print(f"Uploading processed dataset to {output_s3_path}...")
    upload_to_s3(local_output, output_s3_path)
    print(f"Processed data saved successfully at {output_s3_path}")


if __name__ == "__main__":
    # S3 paths for input and output
    input_s3_path = "s3://sagemaker-us-west-1-454001226345/data/raw_data.csv"
    output_s3_path = "s3://sagemaker-us-west-1-454001226345/data/processed_data.csv"

    preprocess_data(input_s3_path, output_s3_path)
