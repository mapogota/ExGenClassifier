import torch
import torch.nn as nn

import numpy as np


def binarize(x):
    if x <= 0:
        return 0
    else:
        return 1
 
    

def discretize(x):
    if x == 0:
        return 0
    elif x <= 0.5:
        return 1
    elif x <= 1.0:
        return 2
    else:
        return 3
    



def generate_synthetic_samples(X, y, target_class, num_samples_needed):
    """
    Generate synthetic samples for a specific class to balance the dataset.

    Parameters:
        X (np.ndarray): Feature matrix (shape: num_samples x num_features)
        y (np.ndarray): Labels corresponding to X
        target_class (int): The class to augment
        num_samples_needed (int): Number of synthetic samples to generate

    Returns:
        np.ndarray: Synthetic feature matrix
        np.ndarray: Corresponding labels for synthetic data
    """
    # Get indices of samples in the target class
    class_indices = np.where(y == target_class)[0]
    class_samples = X[class_indices]

    synthetic_samples = []
    for _ in range(num_samples_needed):
        # Randomly select two samples from the class
        idx1, idx2 = np.random.choice(len(class_samples), size=2, replace=False)
        sample1, sample2 = class_samples[idx1], class_samples[idx2]
        
        # Combine features (90% from one sample and 10% from the other)
        mask = np.random.rand(sample1.shape[0]) < 0.9
        new_sample = np.where(mask, sample1, sample2)
        
        synthetic_samples.append(new_sample)

    synthetic_samples = np.array(synthetic_samples)
    synthetic_labels = np.full((num_samples_needed,), target_class)
    return synthetic_samples, synthetic_labels

def balance_classes(X, y):
    """
    Balance dataset by generating synthetic samples for smaller classes.

    Parameters:
        X (np.ndarray): Feature matrix (shape: num_samples x num_features)
        y (np.ndarray): Labels corresponding to X

    Returns:
        np.ndarray: Augmented feature matrix
        np.ndarray: Augmented labels
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)
    max_count = max(class_counts)

    X_augmented, y_augmented = [X], [y]
    for cls, count in zip(unique_classes, class_counts):
        if count < max_count:
            num_samples_needed = max_count - count
            synthetic_X, synthetic_y = generate_synthetic_samples(X, y, cls, num_samples_needed)
            X_augmented.append(synthetic_X)
            y_augmented.append(synthetic_y)

    return np.vstack(X_augmented), np.hstack(y_augmented)


def make_fc(dims):
    layers = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers[:-1])  # Exclude final ReLU non-linearity


def train(svi, train_loader, use_cuda=False):
    # initialize loss accumulator
    epoch_loss = 0.0
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for x, y in train_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
            y = y.cuda()
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x, y)

    # return epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

@torch.no_grad()
def evaluate(svi, test_loader, use_cuda=False):
    # initialize loss accumulator
    test_loss = 0.0
    # compute the loss over the entire test set
    for x, y in test_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
            y = y.cuda()
        # compute ELBO estimate and accumulate loss
        test_loss += svi.evaluate_loss(x, y)
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test


@torch.no_grad()
def predict(vae, loader, use_cuda=False):
    y_true = torch.tensor([])
    y_preds = torch.tensor([])
    vae.eval()
    for x, y in loader:
        if use_cuda:
            x = x.cuda()
            y = y.cuda()
        z_loc, _ = vae.encoder(x, y)
        _, y_probs = vae.decoder(z_loc)
        y_true = torch.concat([y_true, y.detach().cpu()], dim=0)
        y_preds = torch.concat([y_preds, y_probs.detach().cpu()], dim=0)
    vae.train()
    return y_true, y_preds


@torch.no_grad()
def get_embeddings(vae, loader, use_cuda=False):
    vae.eval()
    X = torch.tensor([])
    XH = torch.tensor([])
    Y = torch.tensor([])
    YH = torch.tensor([])
    Z = torch.tensor([])
    for x, y in loader:
        if use_cuda:
            x = x.cuda()
            y = y.cuda()
        z, _ = vae.encoder(x, y)
        xh, yh = vae.decoder(z)
        X = torch.concat([X, x.detach().cpu()], dim=0)
        Y = torch.concat([Y, y.detach().cpu()], dim=0)
        XH = torch.concat([XH, xh.detach().cpu()], dim=0)
        YH = torch.concat([YH, yh.detach().cpu()], dim=0)
        Z = torch.concat([Z, z.detach().cpu()], dim=0)
        
    return X, Y, XH, YH, Z