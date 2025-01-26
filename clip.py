
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import umap
import matplotlib.pyplot as plt

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Set up a random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# 2. Define a custom Dataset for paired data
class PairedDataset(Dataset):
    def __init__(self, Xp, Xc):
        self.Xp = torch.tensor(Xp, dtype=torch.float32)
        self.Xc = torch.tensor(Xc, dtype=torch.float32)

    def __len__(self):
        return len(self.Xp)

    def __getitem__(self, idx):
        return self.Xp[idx], self.Xc[idx]

# 3. Data augmentations
class ProteomicsAugmentation(nn.Module):
    def __init__(self, noise_std=0.01):
        super().__init__()
        self.noise_std = noise_std

    def forward(self, x):
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

class ClinicalAugmentation(nn.Module):
    def __init__(self, noise_std=0.01):
        super().__init__()
        self.noise_std = noise_std

    def forward(self, x):
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

# 4. Define a flexible encoder with residual connections
class FlexibleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, residual=False):
        super().__init__()
        self.residual = residual
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual:
            out = x
            for layer in self.model:
                if isinstance(layer, nn.Linear) and out.shape[-1] == layer.out_features:
                    out = out + layer(out)
                else:
                    out = layer(out)
            return out
        else:
            return self.model(x)

# 5. Projection head (shared latent space)
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# 6. Define NT-Xent loss
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, z1, z2):
        batch_size = z1.size(0)

        # Normalize embeddings
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)

        # Positive similarities
        positive_sim = self.cosine_similarity(z1, z2)

        # All similarities
        similarities = torch.cat([z1, z2], dim=0) @ torch.cat([z1, z2], dim=0).T
        mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
        similarities = similarities[mask].view(2 * batch_size, -1)

        # Positive logits
        positive_logits = torch.cat([positive_sim, positive_sim], dim=0)

        # Compute loss
        loss = -torch.log(torch.exp(positive_logits / self.temperature) /
                          torch.exp(similarities / self.temperature).sum(dim=1))
        return loss.mean()

# 7. Training pipeline
def train_model(Xp, Xc, epochs=50, batch_size=32, learning_rate=1e-3, latent_dim=10):
    # Prepare data
    dataset = PairedDataset(Xp, Xc)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define models
    proteomics_encoder = FlexibleEncoder(input_dim=Xp.shape[1], hidden_dim=128, output_dim=latent_dim, num_layers=3, residual=True).to(device)
    clinical_encoder = FlexibleEncoder(input_dim=Xc.shape[1], hidden_dim=64, output_dim=latent_dim, num_layers=3, residual=True).to(device)
    projection_head = ProjectionHead(input_dim=latent_dim, output_dim=latent_dim, num_layers=2, hidden_dim=latent_dim).to(device)

    # Loss and optimizer
    criterion = NTXentLoss().to(device)
    optimizer = optim.Adam(
        list(proteomics_encoder.parameters()) +
        list(clinical_encoder.parameters()) +
        list(projection_head.parameters()),
        lr=learning_rate
    )

    # Training loop
    for epoch in range(epochs):
        proteomics_encoder.train()
        clinical_encoder.train()
        projection_head.train()

        epoch_loss = 0

        for batch in dataloader:
            Xp_batch, Xc_batch = batch
            Xp_batch = Xp_batch.to(device)
            Xc_batch = Xc_batch.to(device)

            # Augment data
            Xp_augmented = ProteomicsAugmentation()(Xp_batch)
            Xc_augmented = Xc_batch#ClinicalAugmentation()(Xc_batch)

            # Encode
            z1 = proteomics_encoder(Xp_augmented)
            z2 = clinical_encoder(Xc_augmented)

            # Project to latent space
            z1_proj = projection_head(z1)
            z2_proj = projection_head(z2)

            # Compute loss
            loss = criterion(z1_proj, z2_proj)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

    return proteomics_encoder, clinical_encoder, projection_head

# 8. Visualization of embedding space using UMAP
# def plot_embeddings(proteomics_encoder, clinical_encoder, Xp, Xc, y, save_path='umap_embeddings.png'):
#     # Set models to evaluation mode
#     proteomics_encoder.eval()
#     clinical_encoder.eval()

#     # Generate embeddings
#     with torch.no_grad():
#         z1 = proteomics_encoder(torch.tensor(Xp, dtype=torch.float32).to(device))
#         z2 = clinical_encoder(torch.tensor(Xc, dtype=torch.float32).to(device))

#    # Concatenate embeddings along feature dimension
#     embeddings = torch.cat([z1, z2], dim=1).cpu().numpy()  # Correct concatenation
#     labels = y  # Use the original labels without duplication

#     # Use UMAP to reduce dimensions to 2D
#     reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
#     embeddings_2d = reducer.fit_transform(embeddings)

#     # Plot the embeddings
#     plt.figure(figsize=(10, 8))
#     for label in np.unique(labels):
#         idx = labels == label
#         plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=f'Class {label}', alpha=0.7)
#     plt.title('2D UMAP Embedding Space')
#     plt.xlabel('UMAP1')
#     plt.ylabel('UMAP2')
#     plt.legend()
#     plt.grid(True)

#     # Save the plot
#     plt.savefig(save_path)
#     print(f"Embedding plot saved to {save_path}")
# 8. Visualization of embedding space using UMAP (for continuous labels)
def plot_embeddings(proteomics_encoder, clinical_encoder, Xp, Xc, y, save_path='umap_embeddings.png'):
    # Set models to evaluation mode
    proteomics_encoder.eval()
    clinical_encoder.eval()

    # Generate embeddings
    with torch.no_grad():
        z1 = proteomics_encoder(torch.tensor(Xp, dtype=torch.float32).to(device))
        z2 = clinical_encoder(torch.tensor(Xc, dtype=torch.float32).to(device))

    # Concatenate embeddings along feature dimension
    embeddings = torch.cat([z1, z2], dim=1).cpu().numpy()  # Correct concatenation
    labels = y  # Use the original labels without duplication

    # Use UMAP to reduce dimensions to 2D
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.0, metric='euclidean')
    embeddings_2d = reducer.fit_transform(embeddings)

    # Plot the embeddings with a continuous color scale
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Continuous Labels')
    plt.title('2D UMAP Embedding Space')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.grid(True)

    # Save the plot
    plt.savefig(save_path)
    print(f"Embedding plot saved to {save_path}")



# Run model and visualize embeddings

from data import X_train, X_test, y_train, y_test, xc_train, xc_test

xp_train = X_train
xp_test = X_test
y_train = y_train.values #argmax(axis=-1)
y_test = y_test.values #argmax(axis=-1)

# import pdb; pdb.set_trace()

proteomics_encoder, clinical_encoder, projection_head = train_model(xp_train, xc_train, epochs=1000)

plot_embeddings(proteomics_encoder, clinical_encoder, xp_train, xc_train, y_train, save_path='umap_train.png')
plot_embeddings(proteomics_encoder, clinical_encoder, xp_test, xc_test, y_test, save_path='umap_test.png')

