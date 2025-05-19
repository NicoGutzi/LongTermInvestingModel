from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── Dataset Wrapper ────────────────────────────────────────────────────────────
class TabularDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        # Expect df indexed by date, columns = price + indicators
        # Convert to float32 tensor
        self.X = torch.tensor(df.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # For autoencoder, input==target
        return self.X[idx], self.X[idx]

# ─── Autoencoder Definition ────────────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 5):
        super().__init__()
        # Encoder: input_dim → 128 → latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        # Decoder: latent_dim → 128 → input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

# ─── Training Function ─────────────────────────────────────────────────────────
def train_autoencoder(
    df: pd.DataFrame,
    latent_dim: int = 5,
    batch_size: int = 32,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> pd.DataFrame:

    # Prepare DataLoader
    dataset = TabularDataset(df)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Build model, optimizer, loss
    model     = Autoencoder(input_dim=df.shape[1], latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for x, _ in loader:
            x = x.to(device)
            x_hat, _ = model(x)
            loss = criterion(x_hat, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        epoch_loss = running_loss / len(dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} — Loss: {epoch_loss:.6f}")

    # Extract latent features for all data
    model.eval()
    with torch.no_grad():
        all_X = torch.tensor(df.values, dtype=torch.float32).to(device)
        _, Z   = model(all_X)
        Z      = Z.cpu().numpy()

    # Build a DataFrame of latent vectors
    latent_cols = [f"ae_feat_{i+1}" for i in range(latent_dim)]
    df_latent = pd.DataFrame(Z, index=df.index, columns=latent_cols)

    return df_latent, model

# ─── Main Entry Point ──────────────────────────────────────────────────────────
def unsupervised_learning(data: pd.DataFrame) -> pd.DataFrame:
    latent_df, model = train_autoencoder(
        data,
        latent_dim=5,    # tuneable
        epochs=100,      # more epochs for better convergence
        batch_size=64,
        lr=1e-3
    )

    combined_data = data.join(latent_df, how="inner")

    return combined_data

