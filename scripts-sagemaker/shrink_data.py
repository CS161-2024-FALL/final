import pandas as pd
import boto3
import os


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


def shrink_dataset(input_s3_path, output_s3_path, num_samples=40000):
    """
    Shrink a dataset by sampling a fixed number of rows.

    Args:
        input_s3_path (str): S3 path to the input dataset.
        output_s3_path (str): S3 path to save the shrunk dataset.
        num_samples (int): Number of samples to retain.

    Returns:
        None
    """
    # Temporary local paths
    local_input = "/tmp/raw_data.csv"
    local_output = "/tmp/shrunk_data.csv"

    # Download the dataset from S3
    print(f"Downloading raw dataset from {input_s3_path}...")
    download_from_s3(input_s3_path, local_input)

    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(local_input)

    # Sample N rows
    print(f"Sampling {num_samples} rows from the dataset...")
    df_shrunk = df.sample(n=num_samples, random_state=42)

    # Save the shrunk dataset locally
    print(f"Saving shrunk dataset locally at {local_output}...")
    df_shrunk.to_csv(local_output, index=False)

    # Upload the shrunk dataset to S3
    print(f"Uploading shrunk dataset to {output_s3_path}...")
    upload_to_s3(local_output, output_s3_path)
    print(f"Shrunk dataset successfully saved to {output_s3_path}")


if __name__ == "__main__":
    # S3 paths for input and output
    input_s3_path = "s3://sagemaker-us-west-1-454001226345/data/test_data_ads.csv"
    output_s3_path = "s3://sagemaker-us-west-1-454001226345/data/raw_data.csv"

    # Number of rows to sample
    num_samples = 40000

    shrink_dataset(input_s3_path, output_s3_path, num_samples)
