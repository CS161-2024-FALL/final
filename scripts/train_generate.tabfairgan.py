import pandas as pd
import numpy as np
import pickle
from tabfairgan import TFG

# input and output size
SUBSET = 5000 # input
N_OUT = 50000 # output
EPOCHS = 50
BATCH_SIZE = 256

################################################### EDITED_START - Richard ########################################################
# increase label=1 by downsampling negative label
def increase_rel_freq(df, column, value, rel_freq, seed=None):
    if seed is not None:
        np.random.seed(seed)

    df_value = df[df[column] == value]
    df_others = df[df[column] != value]

    n_value = len(df_value)
    n_downsampled = int(n_value * (1-rel_freq)/rel_freq)

    if(n_downsampled > len(df_others)):
      print("WARNING: data already exceeds ratio. Returning original df")
      return df
    # downsample
    sampled_others = df_others.sample(n=n_downsampled, random_state=seed)
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

df = increase_rel_freq(df, 'label', '1', 0.1, seed=1)

################################################### EDITED_END - Richard ########################################################

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