import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


def extract_metadata(file_path):
  """Extracts metadata such as ages, genders, and tissue types from a series matrix file."""
  metadata_lines = []
  with open(file_path, "rt") as f:
    for line in f:
      if line.startswith("!"):
        metadata_lines.append(line)

  characteristics_lines = [line for line in metadata_lines if "!Sample_characteristics_ch1" in line]

  ages, genders, tissues = [], [], []
  seen_metadata = defaultdict(set)

  match file_path:
    case "data/GSE15745-GPL8490_series_matrix.txt":
      for line in characteristics_lines:
        items = line.split("\t")[1:]
        if "age (y):" in line:
          ages = [float(item.split('"age (y): ')[1].strip('"\n"')) for item in items if '"age (y):' in item]
        if "gender:" in line:
          genders = [item.split('"gender: ')[1].strip('"\n"') for item in items if '"gender:' in item]
          genders = ["M" if g == "Male" else "F" for g in genders]
        if "tissue:" in line:
          tissues = [item.split('"tissue: ')[1].strip('"\n"') for item in items if '"tissue:' in item]

    case "data/GSE40279_series_matrix.txt":
      for line in characteristics_lines:
        items = line.split("\t")[1:]
        if "age (y):" in line:
          ages = [float(item.split('"age (y): ')[1].strip('"\n"')) for item in items if '"age (y):' in item]
        if "gender:" in line:
          genders = [item.split('"gender: ')[1].strip('"\n"') for item in items if '"gender:' in item]
        if "tissue:" in line:
          tissues = [item.split('"tissue: ')[1].strip('"\n"') for item in items if '"tissue:' in item]
      
    case "data/GSE41169_series_matrix.txt":
      for line in characteristics_lines:
        items = line.split("\t")[1:]
        if "age:" in line:
          ages = [float(item.split('"age: ')[1].strip('"\n"')) for item in items if '"age:' in item]
        if "gender:" in line:
          genders = [item.split('"gender: ')[1].strip('"\n"') for item in items if '"gender:' in item]
          genders = ["M" if g == "Male" else "F" for g in genders]
      tissues = ["whole blood"]*len(ages)

    case "data/GSE36064_series_matrix.txt":
      for line in characteristics_lines:
        items = line.split("\t")[1:]
        if "age at collection months:" in line:
          ages = [float(item.split('"age at collection months: ')[1].strip('"\n"')) / 12 for item in items if '"age at collection months:' in item]
        if "gender:" in line:
          genders = [item.split('"gender: ')[1].strip('"\n"') for item in items if '"gender:' in item]
      tissues = ["leukocyte"]*len(ages)
    
    case "data/GSE41037_series_matrix.txt":
      for line in characteristics_lines:
        items = line.split("\t")[1:]
        if "age:" in line:
          ages = [float(item.split('"age: ')[1].strip('"\n"')) for item in items if '"age:' in item]
        if "gender:" in line:
          genders = [item.split('"gender: ')[1].strip('"\n"') for item in items if '"gender:' in item]
          genders = ["M" if g == "male" else "F" for g in genders]
        if "tissue:" in line:
          tissues = [item.split('"tissue: ')[1].strip('"\n"') for item in items if '"tissue:' in item]

    case "data/GSE41826_series_matrix.txt":
      for line in characteristics_lines:
        items = line.split("\t")[1:]
        if "age:" in line:
          ages = [float(item.split('"age: ')[1].strip('"\n"')) for item in items if '"age:' in item]
        if "Sex:" in line:
          genders = [item.split('"Sex: ')[1].strip('"\n"') for item in items if '"Sex:' in item]
          genders = ["M" if g == "Male" else "F" for g in genders]
      tissues = ["frontal_cortex"]*len(ages)

  print(f"Found the following ages: {ages}\ngenders: {genders}\ntissues: {tissues}\nfrom the file: {file_path}\n")
  assert len(ages) != 0 and len(genders) != 0 and len(tissues) != 0
  return ages, genders, tissues

def filter_methylation_data(file_path, cpg_sites):
  methylation_data = pd.read_csv(file_path, sep="\t", comment="!", skiprows=1, index_col=0)
  methylation_data = methylation_data.loc[~methylation_data.index.str.startswith("!"), :]
  return methylation_data.loc[methylation_data.index.intersection(cpg_sites), :]

def preprocess_data(data_dir, cpg_file, output_file):
  methylation_sites = pd.read_csv(cpg_file)
  methylation_sites = list(methylation_sites.iloc[3:, 0])

  combined_data = []
  tissue_types = set()

  for file_name in os.listdir(data_dir):
    if file_name.endswith("_series_matrix.txt"):
      file_path = os.path.join(data_dir, file_name)
      
      ages, genders, tissues = extract_metadata(file_path)
      tissue_types.update(tissues)
      
      methylation_data = filter_methylation_data(file_path, methylation_sites)
      
      for idx, (age, gender, tissue) in enumerate(zip(ages, genders, tissues)):
        sample_data = methylation_data.iloc[:, idx].copy()
        sample_data['Gender'] = 0 if gender == 'M' else 1
        sample_data['Tissue'] = tissue
        sample_data['Age'] = age
        combined_data.append(sample_data)

  final_dataset = pd.concat(combined_data, axis=1).T
  
  encoder = OneHotEncoder()
  tissue_encoded = encoder.fit_transform(final_dataset[['Tissue']]).toarray()
  tissue_columns = encoder.get_feature_names_out(['Tissue'])
  
  tissue_df = pd.DataFrame(tissue_encoded, columns=tissue_columns, index=final_dataset.index)

  final_dataset = final_dataset.drop(columns=['Tissue'])
  final_dataset = pd.concat([final_dataset, tissue_df], axis=1)

  final_dataset.to_csv(output_file, index=False)
  print(final_dataset.head())
  print(final_dataset.shape)

class MethylationDataset(Dataset):
  def __init__(self, dataframe, transform=None):
    self.dataframe = dataframe
    self.transform = transform
    self.features = dataframe.drop(columns=['Age']).values.astype('float32')
    self.labels = dataframe[['Age']].values.astype('float32')

  def __len__(self):
    return len(self.features)

  def __getitem__(self, idx):
    data = self.features[idx]
    label = self.labels[idx]
    if self.transform:
      data = self.transform(data)
    return torch.tensor(data), torch.tensor(label)

class EpigeneticClockNN(nn.Module):
  def __init__(self):
    super(EpigeneticClockNN, self).__init__()
    self.network = nn.Sequential(
        nn.Linear(360, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1)
    )

  def forward(self, x):
    return self.network(x)

if __name__ == "__main__":
  if not os.path.exists("data/preprocessed_dataset.csv"):
    preprocess_data("data/", "CpGs.csv", "data/preprocessed_dataset.csv")
  dataset = pd.read_csv("data/preprocessed_dataset.csv")
  dataset = dataset.dropna()
  torch_dataset = MethylationDataset(dataset)
  dataloader = DataLoader(torch_dataset, batch_size=16, shuffle=True)


  model = EpigeneticClockNN()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  num_epochs = 100
  loss_fn = nn.MSELoss()

  epoch_losses = []
  epoch_mae = []

  plt.ion()
  fig, (ax_loss, ax_mae) = plt.subplots(1, 2, figsize=(12, 5))
  ax_loss.set_title("Training Loss")
  ax_loss.set_xlabel("Epoch")
  ax_loss.set_ylabel("Loss")
  ax_mae.set_title("Mean Absolute Error")
  ax_mae.set_xlabel("Epoch")
  ax_mae.set_ylabel("MAE")

  for epoch in range(num_epochs):
    epoch_loss = 0
    mae_sum = 0
    model.train()
    for x, y in dataloader:
      optimizer.zero_grad()
      pred = model(x)
      loss = loss_fn(pred, y)
      loss.backward()
      optimizer.step()
      
      epoch_loss += loss.item()
      mae_sum += torch.sum(torch.abs(pred - y)).item()

    epoch_loss /= len(dataloader.dataset)
    mae = mae_sum / len(dataloader.dataset)

    epoch_losses.append(epoch_loss)
    epoch_mae.append(mae)
    ax_loss.plot(epoch_losses, label="Loss", color="blue")
    ax_mae.plot(epoch_mae, label="MAE", color="orange")
    fig.canvas.draw()
    plt.pause(0.01)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, MAE: {mae:.4f}")

  plt.ioff()
  plt.show()


