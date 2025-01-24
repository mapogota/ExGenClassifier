import torch
import torch.nn as nn


# Step 4: Implement the Neural Networks
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm1d(output_dim) if output_dim > 1 else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.fc(x)
        out = self.norm(out)
        out = self.activation(out)
        if residual.shape == out.shape:
            out += residual
        return out


class FlexibleEncoder(nn.Module):
    def __init__(self, clinical_dim, proteomics_dim, latent_dim, hidden_dims):
        super(FlexibleEncoder, self).__init__()
        self.input_dim = clinical_dim + proteomics_dim
        layers = []
        for h_dim in hidden_dims:
            layers.append(ResidualBlock(self.input_dim, h_dim))
            self.input_dim = h_dim
        self.layers = nn.Sequential(*layers)
        self.fc_mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_std = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, xc, xp):
        x = torch.cat([xc, xp], dim=-1)
        x = self.layers(x)
        z_mean = self.fc_mean(x)
        z_std = torch.exp(self.fc_std(x))  # Outputs the std for Z
        return z_mean, z_std


class FlexibleDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dims):
        super(FlexibleDecoder, self).__init__()
        self.input_dim = latent_dim
        layers = []
        for h_dim in hidden_dims:
            layers.append(ResidualBlock(self.input_dim, h_dim))
            self.input_dim = h_dim
        self.layers = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, z):
        x = self.layers(z)
        logits = self.fc_out(x)
        return logits
