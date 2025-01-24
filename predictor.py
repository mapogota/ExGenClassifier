import pandas as pd
import pyro.distributions as dist
from pyro.infer import Predictive
import pyro
import torch
from sklearn.metrics import accuracy_score

from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer.autoguide import AutoNormal

from models import FlexibleEncoder, FlexibleDecoder


torch.manual_seed(0)

pyro.clear_param_store()

N = 2000
clinical_dim = 20
proteomics_dim = 5000
latent_dim = 10
output_dim = 2


xc = torch.bernoulli(torch.rand(N, clinical_dim))
xp = torch.bernoulli(torch.rand(N, proteomics_dim))
y = torch.nn.functional.one_hot(
    torch.randint(0, output_dim, (N,)), num_classes=output_dim
).float()

# Split into train and test sets
train_size = 1000
test_size = 1000
xc_train, xc_test = xc[:train_size], xc[train_size:]
xp_train, xp_test = xp[:train_size], xp[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# Initialize model
encoder_hidden_dims = [128, 32]
decoder_hidden_dims = [32, 128]

encoder = FlexibleEncoder(clinical_dim, proteomics_dim, latent_dim, encoder_hidden_dims)
decoder = FlexibleDecoder(latent_dim, output_dim, decoder_hidden_dims)


def model(xp, xc, y):
    batch_size = xc.size(0)
    # Use a plate to define the batch size for z
    with pyro.plate("data", size=batch_size):
        # Prior for latent variable Z
        z_mean, z_std = encoder(xc, xp)
        z = pyro.sample("z", dist.Normal(z_mean, z_std).to_event(1))

        # Likelihood for Y given Z
        logits = decoder(z)
        y_dist = dist.OneHotCategorical(logits=logits)
        pyro.sample("y", y_dist, obs=y)


adam_params = {"lr": 0.005, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

auto_guide = AutoNormal(model)

svi = SVI(model, auto_guide, optimizer, loss=Trace_ELBO())

n_steps = 3001

for step in range(n_steps):
    loss = svi.step(xp_train, xc_train, y_train)
    if step % 100 == 0:
        print("Loss: ", loss)


def prepare_results_df(xp, xc, y_true, filename="TrainResults"):
    predictive = Predictive(model, guide=auto_guide, num_samples=500)(xp, xc, None)
    y = predictive["y"].numpy()
    y_mean = y.mean(axis=0)
    y0 = y_mean[:, 0]
    y1 = y_mean[:, 1]
    y_pred = y_mean.argmax(axis=1)

    df = pd.DataFrame(
        {
            "y_true": y_true.numpy().argmax(axis=1),
            "y_pred": y_pred,
            "y0": y0,
            "y1": y1,
        }
    )

    df.to_csv(f"{filename}.csv", index=None)
    return df


def print_metrics(df):
    acc = accuracy_score(y_true=df.y_true.values, y_pred=df.y_pred.values)
    print(f"Accuracy: {acc:.4f}")


train_rdf = prepare_results_df(xp_train, xc_train, y_train, filename="TrainResults")
test_rdf = prepare_results_df(xp_test, xc_test, y_test, filename="TestResults")


print_metrics(train_rdf)
print_metrics(test_rdf)
print("\n\n")
