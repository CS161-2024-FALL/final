import pandas as pd
import pickle
from tabfairgan import TFG

PICKLE_OUT = 'tfg.unfair.pkl'
# Load your dataset
df = pd.read_csv("../../train_merged_output.csv")
df = df[1: 10**5]

# Define fairness configuration
"""
fairness_config = {
    'fair_epochs': 50,
    'lamda': 0.5,
    'S': 'sex',
    'Y': 'income',
    'S_under': ' Female',
    'Y_desire': ' >50K'
}
"""
fairness_config = {}

# Initialize TabFairGAN with fairness constraints
tfg = TFG(df, epochs=200, batch_size=256, device='cuda:0')

# Train the model
tfg.train()

with open(PICKLE_OUT,'wb') as f:
    pickle.dump(tfg,f)

# Generate synthetic data
fake_df = tfg.generate_fake_df(num_rows=32561)
fake_df

fake_df.to_csv('../../synthetic_data.train_merged_output.tabfairgan.csv', index=False)

with open(PICKLE_OUT, 'rb') as f:
    tfg_test = pickle.load(f)