import pandas as pd


def shrink_dataset(input_path, output_path, num_samples=40000):
    # Load raw data
    df = pd.read_csv(input_path)

    # Sample N rows
    df_shrunk = df.sample(n=num_samples, random_state=42)

    # Save the shrunk dataset
    df_shrunk.to_csv(output_path, index=False)
    print(f"Shrunk dataset saved to {output_path}")


if __name__ == "__main__":
    input_file = r"C:\SCHOOL\FALL 24\STATS 161\midterm\test_data_ads.csv"
    output_file = "data/raw_data.csv"
    shrink_dataset(input_file, output_file)
