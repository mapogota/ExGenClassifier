import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from data import NUM_CLASSES, one_hot_encode
from utils import balance_classes

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

class XcPredictor(nn.Module):
    def __init__(self, proteomics_dim, clinical_dim, hidden_dims):
        super(XcPredictor, self).__init__()
        self.input_dim = proteomics_dim
        layers = []
        for h_dim in hidden_dims:
            layers.append(ResidualBlock(self.input_dim, h_dim))
            self.input_dim = h_dim
        self.layers = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dims[-1], clinical_dim)
        self.sigmoid = nn.Sigmoid()  # Output activation to ensure predicted probabilities are between 0 and 1.

    def forward(self, xp):
        x = self.layers(xp)
        xc_prob = self.fc_out(x)
        xc_prob = self.sigmoid(xc_prob)
        return xc_prob

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
    def __init__(self, clinical_dim, proteomics_dim, latent_dim, output_dim, encoder_hidden_dims, decoder_hidden_dims, xc_predictor_hidden_dims):
        super(ProbabilisticModel, self).__init__()
        self.xc_predictor = XcPredictor(proteomics_dim, clinical_dim, xc_predictor_hidden_dims)
        self.encoder = FlexibleEncoder(clinical_dim, proteomics_dim, latent_dim, encoder_hidden_dims)
        self.decoder = FlexibleDecoder(latent_dim, output_dim, decoder_hidden_dims)
        self.latent_dim = latent_dim
        self.clinical_dim = clinical_dim
        self._initialize_weights()

    def _initialize_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
        self.xc_predictor.apply(init_weights)

    def model(self, xc, xp, y=None):
        pyro.module("decoder", self.decoder)
        batch_size = xp.size(0)
        options = dict(dtype=xp.dtype, device=xp.device)
        eps = 1e-4

        with pyro.plate("data", size=batch_size):
            # # Prior for Xc
            # if xc is None: #If we do not pass it a value.
            #     xc_prior_prob = 0.5 * torch.ones(batch_size, self.clinical_dim, **options)
            # else:  #If we do pass it a value
            #    xc_prior_prob = xc + eps

            # xc_prior = dist.Bernoulli(probs=xc_prior_prob, validate_args=False).to_event(1)
            # xc = pyro.sample("xc", xc_prior)

            # Prior for Xc, conditioned on Xp
            # if xc is None:  # If xc is not observed
            xc_prior_prob = 0.5 * torch.ones(batch_size, self.clinical_dim, **options)
            xc_prior = dist.Bernoulli(probs=xc_prior_prob, validate_args=False).to_event(1)
            xc = pyro.sample("xc", xc_prior, obs=xc)
       

            # Prior for latent variable Z
            z_prior = dist.Normal(torch.zeros(batch_size, self.latent_dim, **options),
                                  torch.ones(batch_size, self.latent_dim, **options)).to_event(1)
            z = pyro.sample("z", z_prior)

            # Likelihood for Y given Z
            logits = self.decoder(z)
            y_dist = dist.OneHotCategorical(logits=logits)
            pyro.sample("y", y_dist, obs=y)

    def guide(self, xc, xp, y=None):
        pyro.module("encoder", self.encoder)
        pyro.module("xc_predictor", self.xc_predictor)

        batch_size = xp.size(0)
        options = dict(dtype=xp.dtype, device=xp.device)
        
        with pyro.plate("data", size=batch_size):
            # Approximate posterior for Xc
            xc_prob = self.xc_predictor(xp)
            xc_dist = dist.Bernoulli(probs=xc_prob).to_event(1)
            xc = pyro.sample("xc", xc_dist, infer={"is_auxiliary": True})

            # Approximate posterior for Z
            z_mean, z_std = self.encoder(xc, xp) #Use predicted xc here
            z_dist = dist.Normal(z_mean, z_std).to_event(1)
            pyro.sample("z", z_dist)

# Step 7: Set Up the Training Process
def train(model, dataloader, num_epochs=100, learning_rate=1e-3, device='cpu'):
    model = model.to(device)
    loss_list = []
    acc_list = []
    svi = SVI(model.model, model.guide, Adam({"lr": learning_rate}), loss=Trace_ELBO())
    for epoch in range(num_epochs):
        epoch_loss = 0
        for xc, xp, y in dataloader:
            xc, xp, y = xc.to(device), xp.to(device), y.to(device)
            loss = svi.step(xc, xp, y)
            epoch_loss += loss
        avg_loss = epoch_loss / len(dataloader.dataset)
        loss_list.append(avg_loss)
        acc = test(model, test_dataloader, device=device, print_acc=False)
        acc_list.append(acc)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Valid Acc: {acc:.4f}")
    return loss_list, acc_list

def test(model, dataloader, device='cpu', print_acc=True):
    model.to(device)
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for xc, xp, y in dataloader:
            xc, xp, y = xc.to(device), xp.to(device), y.to(device)
            xc_prob = model.xc_predictor(xp)
            xc = xc_prob #dist.Bernoulli(probs = xc_prob).sample() # Sample during test time
            z_mean, _ = model.encoder(xc, xp)
            logits = model.decoder(z_mean)
            predictions = torch.argmax(logits, dim=-1)
            targets = torch.argmax(y, dim=-1)
            total += y.size(0)
            correct += (predictions == targets).sum().item()
    accuracy = correct / total
    if print_acc:
        print(f"Test Accuracy: {accuracy:.4f}")
    model.train()
    return accuracy

# Example Usage
if __name__ == "__main__":
    # Dataset
    import config
    from data import X_train, X_test, y_train, y_test, xc_train, xc_test

    torch.manual_seed(0)


    # X_balanced, y_balanced = balance_classes(X_train, y_train)

    # import pdb; pdb.set_trace()
    # X_train = X_balanced
    # y_train = one_hot_encode(y_balanced, NUM_CLASSES)

    xp_train, xp_test = torch.from_numpy(X_train).float(), torch.from_numpy(X_test).float()
    xc_train, xc_test = torch.from_numpy(xc_train).float(), torch.from_numpy(xc_test).float()
    y_train, y_test = torch.from_numpy(y_train).float(), torch.from_numpy(y_test).float()
    
    # N = X_train.shape[0] + X_test.shape[0]    
    proteomics_dim = X_train.shape[-1]
    clinical_dim = xc_train.shape[-1]
    output_dim = y_train.shape[-1]
    latent_dim = 10

    # print(xc_train.shape)
    # import sys; sys.exit()
    
    

    # Create DataLoaders
    train_dataset = TensorDataset(xc_train, xp_train, y_train)
    test_dataset = TensorDataset(xc_test, xp_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Initialize model
    encoder_hidden_dims = [512, 128, 64]
    decoder_hidden_dims = [64, 128, 512]
    xc_predictor_hidden_dims = [64, 128]
    model = ProbabilisticModel(clinical_dim, proteomics_dim, latent_dim, output_dim, encoder_hidden_dims, decoder_hidden_dims, xc_predictor_hidden_dims)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move model to the device
    model = model.to(device) 
   

    # Train the model
    elbo_list, acc_list = train(model, train_dataloader, num_epochs=config.NUM_EPOCHS, learning_rate=config.LEARNING_RATE, device=device)

    # Test the model
    test(model, test_dataloader, device=device)

    baseline_acc = (y_test.sum(dim=0)/y_test.shape[0]).max().item()
    print(f"Baseline classification accuracy: {baseline_acc:.4f}")

    fig, ax = plt.subplots(nrows=2, ncols=1)
    

    start_pos = 0
    # Plot ELBO Loss
    ax[0].plot(elbo_list[start_pos:])
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('ELBO Loss (Train)')
    

    # Plot ELBO Loss
    ax[1].plot(acc_list[start_pos:])
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy (Validation)')
    ax[1].set_ylim([0, 1])

    fig.subplots_adjust(hspace=0.5)
    fig.suptitle('Learning curves')
    plt.show()