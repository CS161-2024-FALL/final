# Tried to get this working, but tests kept getting stuck, so we abandon
# the usage of sage maker, but here was the code that we got furthest with

import boto3
import sagemaker
from sagemaker.remote_function import remote

# Initialize SageMaker session
sm_session = sagemaker.Session(
    boto_session=boto3.session.Session(region_name="us-west-1")
)

# These settings are what we used to config the sagemaker job
# this is a custom built role that i made, you will also need
# to set up AWS IAM if you want to run this
settings = dict(
    sagemaker_session=sm_session,
    # This is a custom role i made, you will need to make your own
    role="arn:aws:iam::454001226345:role/SageMakerFullAccessRole",
    # We used this instance because AWS offered free hours
    instance_type="ml.m4.xlarge",
    # Of course, we need requirements so the image can be the same
    dependencies="./requirements.txt",
)


# This annotation is specific to Sagemaker.
# The Docs: https://docs.aws.amazon.com/sagemaker/latest/dg/train-remote-decorator.html
@remote(**settings)
def generate_synthetic_data_remote(
    input_path, output_path, model_path, n_sample, fairness_config=None
):
    import pandas as pd
    import pickle
    from tabfairgan import TFG

    # loading....
    print(f"Loading processed dataset from {input_path}...")
    df = pd.read_csv(input_path)

    # deciding which columns to use and bucketing them
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
    ]

    # we do convert to strings so that TabFairGAN can process them
    for col in categorical_columns:
        df[col] = df[col].astype(str)

    # remove data we don't need
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

    # we do init the model
    print("Initializing TabFairGAN...")
    tfg = TFG(
        df,
        epochs=200,
        batch_size=256,
        # you would seet this to cuda if you have a GPU
        device="cpu",
        fairness_config=fairness_config,
    )

    # training.... takes a long time
    print("Training TabFairGAN...")
    tfg.train()

    # so we don't lose the model, we save it
    print(f"Saving TabFairGAN model to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(tfg, f)

    # we generate the data and save it
    print(f"Generating {n_sample} synthetic rows with TabFairGAN...")
    fake_df = tfg.generate_fake_df(num_rows=n_sample)
    print(f"Saving synthetic dataset to {output_path}...")
    fake_df.to_csv(output_path, index=False)
    print(f"Synthetic data saved successfully at {output_path}")


if __name__ == "__main__":

    # S3 paths for input/output
    # note you will need to set this up yourself in S3...
    input_path = "s3://sagemaker-us-west-1-454001226345/data/raw_data.csv"
    output_path = "s3://sagemaker-us-west-1-454001226345/output/synthetic_data.csv"
    model_path = "s3://sagemaker-us-west-1-454001226345/output/tabfairgan_model.pkl"

    # num of synthetic rows to generate
    n_sample = 10000

    # our fairness configuration
    fairness_config = {
        "fair_epochs": 50,
        "lamda": 0.5,
        "S": "gender",
        "Y": "city_rank",
        "S_under": "3",
        "Y_desire": ">6",
    }

    # we run it up
    generate_synthetic_data_remote(
        input_path, output_path, model_path, n_sample, fairness_config
    )
