import pandas as pd


RAW_F = r"C:\SCHOOL\FALL 24\STATS 161\midterm\test_data_ads.csv"
OUTPUT_F = "data/data_ads_100k.csv"
N_SAMPLES = 100000


def shrink_dataset(input_path, output_path):
    # this is raw raw fr ft data
    df = pd.read_csv(input_path)

    # Sample N rows
    df_shrunk = df.sample(n=N_SAMPLES, random_state=110)

    # Save the shrunk dataset
    df_shrunk.to_csv(output_path, index=False)
    print(f"Shrunk dataset saved to {output_path}")


if __name__ == "__main__":
    input_file = RAW_F
    output_file = "data/raw_data.csv"
    shrink_dataset(input_file, output_file)
