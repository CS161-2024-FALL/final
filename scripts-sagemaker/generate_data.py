# Script for training GAN and generating synthetic data
import pandas as pd
from tabgan.sampler import GANGenerator
from utils import prepare_dataset
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


def generate_synthetic_data(input_s3_path, output_s3_path, n_sample):
    """
    Generate synthetic data using TabGAN.

    Args:
        input_s3_path (str): S3 path to the preprocessed dataset CSV.
        output_s3_path (str): S3 path to save the synthetic dataset CSV.
        n_sample (int): Number of synthetic rows to generate.

    Returns:
        None
    """
    # Temporary local paths
    local_input = "/tmp/processed_data.csv"
    local_output = "/tmp/synthetic_data.csv"

    # Download the dataset from S3
    print(f"Downloading processed dataset from {input_s3_path}...")
    download_from_s3(input_s3_path, local_input)

    # Load the processed dataset
    print(f"Loading processed dataset...")
    df = pd.read_csv(local_input)

    # Prepare dataset for TabGAN
    print("Preparing dataset for TabGAN...")
    train, target, test = prepare_dataset(df, n_sample)

    # Generate synthetic data
    print(f"Generating synthetic data with TabGAN...")
    generator = GANGenerator()
    new_train, new_target = generator.generate_data_pipe(
        train, target, test, only_generated_data=False
    )

    # Save synthetic data to a local CSV file
    synthetic_data = pd.concat([new_train, new_target], axis=1)
    print(f"Saving synthetic dataset locally at {local_output}...")
    synthetic_data.to_csv(local_output, index=False)

    # Upload the synthetic dataset to S3
    print(f"Uploading synthetic dataset to {output_s3_path}...")
    upload_to_s3(local_output, output_s3_path)
    print(f"Synthetic data successfully saved to {output_s3_path}")


if __name__ == "__main__":
    # S3 paths for input and output
    input_s3_path = "s3://sagemaker-us-west-1-454001226345/data/processed_data.csv"
    output_s3_path = "s3://sagemaker-us-west-1-454001226345/data/synthetic_data.csv"

    # Specify the number of synthetic rows to generate
    n_sample = 10000

    generate_synthetic_data(input_s3_path, output_s3_path, n_sample)
