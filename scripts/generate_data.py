# Script for training GAN and generating synthetic data
import pandas as pd
import pickle
from tabgan.sampler import GANGenerator
from utils import prepare_dataset


def generate_synthetic_data(input_path, output_path, model_path, n_sample):
    """
    Generate synthetic data using TabGAN.

    Args:
        input_path (str): Path to the preprocessed dataset CSV.
        output_path (str): Path to save the synthetic dataset CSV.
        n_sample (int): Number of synthetic rows to generate.

    Returns:
        None
    """
    # Load the processed dataset
    print(f"Loading processed dataset from {input_path}...")
    df = pd.read_csv(input_path)

    # Prepare dataset for TabGAN
    print("Preparing dataset for TabGAN...")
    train, target, test = prepare_dataset(df, n_sample)

    # Generate synthetic data
    print(f"Generating {len(train)} synthetic rows with TabGAN...")
    generator = GANGenerator()
    new_train, new_target = generator.generate_data_pipe(
        train, target, test, only_generated_data=False
    )

    # write model (pickle)
    with open(model_file,'wb') as f:
        pickle.dump(generator, f)

    # Save synthetic data to a CSV file
    synthetic_data = pd.concat([new_train, new_target], axis=1)
    print(f"Saving synthetic dataset to {output_path}...")
    synthetic_data.to_csv(output_path, index=False)
    print(f"Synthetic data saved successfully at {output_path}")


if __name__ == "__main__":
    input_file = "data/processed_data.csv"
    output_file = "data/synthetic_data.csv"

    # Specify the number of synthetic rows to generate
    n_sample = 10000

    model_file = f"tabgan.n{n_sample}.pkl"

    generate_synthetic_data(input_file, output_file, model_file, n_sample)
