import pandas as pd
import numpy as np
import pickle
from tabfairgan import TFG

# input and output size
SUBSET = 5000 # input
N_OUT = 50000 # output
EPOCHS = 50
BATCH_SIZE = 256

# increase label=1 by downsampling negative label
def upsample_df(df, column, value, ratio, seed=None):
    if seed is not None:
        np.random.seed(seed)

    df_value = df[df[column] == value]
    df_others = df[df[column] != value]

    n_value = len(df_value)
    n_others = int(n_value / ratio) - n_value if ratio < 1 else int(n_value * ratio) - n_value

    if n_others > len(df_others):
        raise ValueError("Not enough data to achieve desired ratoi.")
    

    # downsample
    sampled_others = df_others.sample(n=n_others, random_state=seed)
    sampled_df = pd.concat([df_value, sampled_others]).sample(frac=1, random_state=seed).reset_index(drop=True)

    return sampled_df

# model hyperparameters
LAMDA = 0.5 # tradeoff between fairness and accuracy
S = 'age'
S_UNDER = '4'
Y = 'label'
Y_DESIRE = '1'

# Load your dataset
df = pd.read_csv("train_merged_output.csv", nrows=SUBSET*10)
df['label'] = df['label'].astype(str)
df['cillabel'] = df['cillabel'].astype(str)
df['age'] = df['age'].astype(str)

df = upsample_df(df, 'label', '1', 0.1, seed=1)

# Define fairness configuration
fairness_config = {
    'fair_epochs': 50,
    'lamda': LAMDA, # tradeoff between fairness and accuracy. this is how they spelled it lol
    'S': S,
    'Y': Y,
    'S_under': S_UNDER, # underrepresented group
    'Y_desire': Y_DESIRE # desired outcome
}

# Initialize TabFairGAN with fairness constraints
tfg = TFG(df, epochs=EPOCHS, batch_size=BATCH_SIZE, device='cuda:0', fairness_config=fairness_config)
model_subname = f'fair.{S}.{LAMDA}' if fairness_config else 'unfair'
PICKLE_OUT = f'tabfairgan.{model_subname}.pkl'

# Train the model
tfg.train()

with open(PICKLE_OUT,'wb') as f:
    pickle.dump(tfg,f)

# Generate synthetic data
fake_df = tfg.generate_fake_df(num_rows=N_OUT)
fake_df

fake_df.to_csv(f'synthetic_data.train_merged_output.tabfairgan.{model_subname}.csv', index=False)

with open(PICKLE_OUT, 'rb') as f:
    tfg_test = pickle.load(f)