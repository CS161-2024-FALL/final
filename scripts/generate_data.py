import pandas as pd
import numpy as np
import pickle
from tabfairgan import TFG


# TRAINING INPOUT OUTPUT SIZE CONFIGS
SUBSET = 5000
N_OUT = 50000
EPOCHS = 100
BATCH_SIZE = 256
DEVICE = "cuda:0"  # can use gpu by setting to "cuda:0"


# FILE PATHS
INPUT_F = "data/data_ads_500k.csv"
OUTPUT_F = "data/synthetic_fair_data_ads_500k.csv"
MODEL_F = "data/tabgan_fair_ads_500k_model.pkl"


# FAIRNESS CONFIG
# note: tradeoff between fairness and accuracy
# FAIR_EPOCHS = 50
# LAMDA = 0.5
# S = "age"
# S_UNDER = "4"
# Y = "label"
# Y_DESIRE = "1"

# CONIGS for GENDER
FAIR_EPOCHS = 50
LAMDA = 0.5
S = "gender"
S_UNDER = "2"
Y = "label"
Y_DESIRE = "0"


# SOME UTIL FUNCTIONS


# increase label=1 by downsampling negative label
def increase_rel_freq(df, column, value, rel_freq, seed=None):
    if seed is not None:
        np.random.seed(seed)

    df_value = df[df[column] == value]
    df_others = df[df[column] != value]

    n_value = len(df_value)
    n_downsampled = int(n_value * (1 - rel_freq) / rel_freq)

    if n_downsampled > len(df_others):
        print("WARNING: Data already exceeds ratio. Returning original df.")
        return df

    sampled_others = df_others.sample(n=n_downsampled, random_state=seed)
    sampled_df = (
        pd.concat([df_value, sampled_others])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    return sampled_df


def generate_synthetic_data(
    input_path, output_path, model_path, n_sample, fairness_config=None
):

    # we load in preprocessed data (the data we shrunk down)
    print(f"Loading processed dataset from {input_path}...")
    df = pd.read_csv(input_path)

    # we increase the relative frequency of label=1 by downsampling negative label
    print("Increasing relative frequency of label=1 by downsampling negative label...")
    df = increase_rel_freq(df, "label", 1, 0.1, seed=1)

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
        "age",
    ]
    for col in categorical_columns:
        df[col] = df[col].astype(str)

    # we remove columns we don't need
    columns_to_remove = [
        "ad_click_list_v001",
        "ad_click_list_v002",
        "ad_click_list_v003",
        "ad_close_list_v001",
        "ad_close_list_v002",
        "ad_close_list_v003",
        "u_newsCatInterestsST",
    ]
    df = df.drop(columns=columns_to_remove)

    # we init the model
    print("Initializing TabFairGAN...")
    tfg = TFG(
        df,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        device="cpu",  # Use GPU if available
        fairness_config=fairness_config,  # Apply fairness constraints if provided
    )

    # DEBUGGING so we can see the data
    print("Dataset Cols and Types:")
    print(tfg.df.dtypes)

    print("\nFirst 5 Rows of Dataset:")
    print(tfg.df.head())

    # more debugging to see tabfairgan detected columns
    print("Discrete Columns Identified by TabFairGAN:")
    print(tfg.df.select_dtypes(include=["object", "category"]).columns.tolist())

    # the actual training happens... very long time
    print("Training TabFairGAN...")
    tfg.train()

    # we save the model
    print(f"Saving TabFairGAN model to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(tfg, f)

    # we generate the synthetic data
    print(f"Generating {n_sample} synthetic rows with TabFairGAN...")
    fake_df = tfg.generate_fake_df(num_rows=n_sample)

    # we save the synthetic dataset to a CSV file
    print(f"Saving synthetic dataset to {output_path}...")
    fake_df.to_csv(output_path, index=False)
    print(f"Synthetic data saved successfully at {output_path}")


if __name__ == "__main__":
    # connecting our configs
    input_file = INPUT_F
    output_file = OUTPUT_F
    model_file = MODEL_F
    n_sample = N_OUT
    fairness_config = {
        "fair_epochs": FAIR_EPOCHS,
        "lamda": LAMDA,
        "S": S,
        "Y": Y,
        "S_under": S_UNDER,
        "Y_desire": Y_DESIRE,
    }

    # we run the function
    generate_synthetic_data(
        input_file, output_file, model_file, n_sample, fairness_config
    )
