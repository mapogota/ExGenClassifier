import os
import os.path as osp
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import config
from utils import binarize

BASE_DIR = os.getcwd()
DATA_FILE = osp.join(BASE_DIR, "Exce.csv")
PHENO_FILE = osp.join(BASE_DIR,"COPDGene_P1P2P3_Flat_SM_NS_Sep23interim.parquet")
NUM_CLASSES = 2

cols = [
        "sid_object",
        "gender_category", 
        "Age_P2_float", 
        "BMI_P2_float", 
        "Exacerbation_Frequency_P2_category", 
        "ATS_PackYears_P2_float",
        "smoking_status_P2_category",
        "eosinphl_P2_float",
        "neutr_lymph_ratio_P2_float"
        ]

def load_data(file_path):
    return pd.read_csv(file_path, index_col="Unnamed: 0")

def preprocess_data(df):
    y = df.pop("LFU_Net_Exacerbations_mean")#.apply(lambda x: binarize(x))
    sids = df.pop("SID")
    return df, y, sids

def load_pheno_data(fpath, sids, cols):
    df = pd.read_parquet(fpath)
    df["neutr_lymph_ratio_P2_float"] = df["neutrophl_P2_float"] / df["lymphcyt_P2_float"]
    df = df[cols]
    df = df[df["sid_object"].isin(sids)]
    # df = df.fillna(0.0)
    df = df.reset_index(drop=True)
    return df

def preprocess_pheno_data(df):
    # Placeholder for processed columns
    processed_columns = []

    # Iterate through each column
    for col in df.columns:
        if df[col].dtype == "category":
            # One-hot encode categorical columns
            encoder = OneHotEncoder(sparse_output=False, drop='first')  # Use `sparse_output` instead of `sparse`
            encoded = encoder.fit_transform(df[[col]])
            encoded_df = pd.DataFrame(
                encoded,
                columns=encoder.get_feature_names_out([col]),
                index=df.index
            )
            processed_columns.append(encoded_df)
        elif df[col].dtype == "float64":
            # Normalize float columns
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled = scaler.fit_transform(df[[col]])
            scaled_df = pd.DataFrame(
                scaled,
                columns=[col],
                index=df.index
            )
            processed_columns.append(scaled_df)
        else:
            # Leave other types (e.g., object columns) unchanged
            processed_columns.append(df[[col]])

    # Concatenate all processed columns back together
    processed_df = pd.concat(processed_columns, axis=1)
    del processed_df["sid_object"]
    return processed_df




def split_data(df, y):
    return train_test_split(df, y, test_size=0.2, stratify=None)

def scale_data(train_df, test_df, df):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_df)
    return scaler.transform(train_df), scaler.transform(test_df), scaler.transform(df)

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

def create_tensors(X_train, y_train, X_test, y_test, X, y):
    train_features = torch.tensor(X_train, dtype=torch.float32)
    train_labels = torch.tensor(y_train, dtype=torch.long)
    test_features = torch.tensor(X_test, dtype=torch.float32)
    test_labels = torch.tensor(y_test, dtype=torch.long)
    full_features = torch.tensor(X, dtype=torch.float32)
    full_labels = torch.tensor(y, dtype=torch.long)
    return train_features, train_labels, test_features, test_labels, full_features, full_labels

def create_dataloaders(train_features, train_labels, test_features, test_labels, full_features, full_labels):
    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    full_dataset = TensorDataset(full_features, full_labels)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    full_loader = DataLoader(full_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    return train_loader, test_loader, full_loader


df = load_data(DATA_FILE)
df, y, sids = preprocess_data(df)
pheno_df = load_pheno_data(fpath=PHENO_FILE,sids=sids, cols=cols )
pheno_df = preprocess_pheno_data(pheno_df)

train_df, test_df, y_train, y_test = split_data(df, y)
X_train, X_test, X = scale_data(train_df.values, test_df.values, df.values)
# y_train = one_hot_encode(y_train, NUM_CLASSES)
# y_test = one_hot_encode(y_test, NUM_CLASSES)
# y = one_hot_encode(y, NUM_CLASSES)
# np.nan_to_num(arr, nan=0.0)
xc_train = np.nan_to_num(pheno_df[pheno_df.index.isin(train_df.index)].values, nan=0.0)
xc_test = np.nan_to_num(pheno_df[pheno_df.index.isin(test_df.index)].values, nan=0.0)

# train_features, train_labels, test_features, test_labels, full_features, full_labels = create_tensors(X_train, y_train, X_test, y_test, X, y)
# train_loader, test_loader, full_loader = create_dataloaders(train_features, train_labels, test_features, test_labels, full_features, full_labels)

# for x, y in test_loader:
#     print(x.shape, y.shape)

