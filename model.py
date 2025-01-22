import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

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

# Step 5: Implement the Pyro Model
class ProbabilisticModel(PyroModule):
    def __init__(self, clinical_dim, proteomics_dim, latent_dim, output_dim, encoder_hidden_dims, decoder_hidden_dims):
        super(ProbabilisticModel, self).__init__()
        self.encoder = FlexibleEncoder(clinical_dim, proteomics_dim, latent_dim, encoder_hidden_dims)
        self.decoder = FlexibleDecoder(latent_dim, output_dim, decoder_hidden_dims)
        self.latent_dim = latent_dim
        self._initialize_weights()

    def _initialize_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)

    def model(self, xc, xp, y=None):
        pyro.module("decoder", self.decoder)

        # Use a plate to define the batch size for z
        with pyro.plate("data", size=xc.size(0)):
            # Prior for latent variable Z
            z_prior = dist.Normal(torch.zeros(xc.size(0), self.latent_dim),
                                  torch.ones(xc.size(0), self.latent_dim)).to_event(1)
            z = pyro.sample("z", z_prior)

            # Likelihood for Y given Z
            logits = self.decoder(z)
            y_dist = dist.OneHotCategorical(logits=logits)
            pyro.sample("y", y_dist, obs=y)

    def guide(self, xc, xp, y=None):
        pyro.module("encoder", self.encoder)

        # Use a plate to define the batch size for z
        with pyro.plate("data", size=xc.size(0)):
            # Approximate posterior for Z
            z_mean, z_std = self.encoder(xc, xp)
            z_dist = dist.Normal(z_mean, z_std).to_event(1)
            pyro.sample("z", z_dist)

# Step 7: Set Up the Training Process
def train(model, dataloader, num_epochs=100, learning_rate=1e-3):
    loss_list = []
    svi = SVI(model.model, model.guide, Adam({"lr": learning_rate}), loss=Trace_ELBO())
    for epoch in range(num_epochs):
        epoch_loss = 0
        for xc, xp, y in dataloader:
            loss = svi.step(xc, xp, y)
            epoch_loss += loss
        avg_loss = epoch_loss / len(dataloader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        loss_list.append(avg_loss)
    return loss_list

def test(model, dataloader):
    total = 0
    correct = 0
    with torch.no_grad():
        for xc, xp, y in dataloader:
            z_mean, _ = model.encoder(xc, xp)
            logits = model.decoder(z_mean)
            predictions = torch.argmax(logits, dim=-1)
            targets = torch.argmax(y, dim=-1)
            total += y.size(0)
            correct += (predictions == targets).sum().item()
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

# Example Usage
if __name__ == "__main__":
    # Synthetic data generation (Step 9: Debugging)
    N = 3000
    clinical_dim = 20
    proteomics_dim = 5000
    latent_dim = 10
    output_dim = 4

    torch.manual_seed(0)
    xc = torch.bernoulli(torch.rand(N, clinical_dim))
    xp = torch.bernoulli(torch.rand(N, proteomics_dim))
    y = torch.nn.functional.one_hot(torch.randint(0, output_dim, (N,)), num_classes=output_dim).float()

    # Split into train and test sets
    train_size = 2000
    test_size = 1000
    xc_train, xc_test = xc[:train_size], xc[train_size:]
    xp_train, xp_test = xp[:train_size], xp[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Create DataLoaders
    train_dataset = TensorDataset(xc_train, xp_train, y_train)
    test_dataset = TensorDataset(xc_test, xp_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize model
    encoder_hidden_dims = [512, 128, 64]
    decoder_hidden_dims = [64, 128, 512]
    model = ProbabilisticModel(clinical_dim, proteomics_dim, latent_dim, output_dim, encoder_hidden_dims, decoder_hidden_dims)

    # Train the model
    elbo_list = train(model, train_dataloader, num_epochs=300, learning_rate=1e-3)

    # Test the model
    test(model, test_dataloader)

    # Plot ELBO Loss
    plt.plot(elbo_list[2:])
    plt.xlabel('Epoch')
    plt.ylabel('ELBO Loss')
    plt.title('ELBO Loss over Epochs')
    plt.show()
