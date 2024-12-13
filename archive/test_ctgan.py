import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

print("GOT HERE 1: Loading CSV data")
data = pd.read_csv("test_data_ads.csv").sample(n=1000, replace=True, random_state=42)
print(data.columns)
data = data[["log_id", "user_id", "age", "gender", "city_rank", "u_refreshTimes"]]
print(data.columns)


print("GOT HERE 2: Scaling data")
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
real_data = torch.tensor(data_scaled, dtype=torch.float32)

num_features = real_data.shape[1]
latent_dim = 5
batch_size = 64
num_epochs = 1000
learning_rate = 0.0002

print("GOT HERE 3: Defining Generator model")


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_features),
        )

    def forward(self, z):
        return self.model(z)


print("GOT HERE 4: Defining Discriminator model")


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


print("GOT HERE 5: Initializing models")
generator = Generator()
discriminator = Discriminator()

print("GOT HERE 6: Setting up loss and optimizers")
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

print("GOT HERE 7: Starting training loop")
for epoch in range(num_epochs):
    optimizer_D.zero_grad()
    real_samples = real_data[torch.randint(0, real_data.size(0), (batch_size,))]
    real_labels = torch.ones((batch_size, 1))

    real_outputs = discriminator(real_samples)
    d_loss_real = criterion(real_outputs, real_labels)

    z = torch.randn((batch_size, latent_dim))
    fake_samples = generator(z)
    fake_labels = torch.zeros((batch_size, 1))

    fake_outputs = discriminator(fake_samples.detach())
    d_loss_fake = criterion(fake_outputs, fake_labels)

    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    optimizer_D.step()

    optimizer_G.zero_grad()

    gen_labels = torch.ones((batch_size, 1))
    gen_outputs = discriminator(fake_samples)
    g_loss = criterion(gen_outputs, gen_labels)

    g_loss.backward()
    optimizer_G.step()

    if epoch % 100 == 0:
        print(
            f"GOT HERE 8: Epoch [{epoch}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}"
        )

print("GOT HERE 9: Generating synthetic data")
z = torch.randn((batch_size, latent_dim))
synthetic_data = generator(z).detach().numpy()
synthetic_data_original_scale = pd.DataFrame(scaler.inverse_transform(synthetic_data))

print("GOT HERE 10: Synthetic data in original scale:")
print(synthetic_data_original_scale.round(4))
