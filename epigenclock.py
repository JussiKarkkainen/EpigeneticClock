import pandas as pd
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

def preprocess_data_dir(data_dir, cpg_file, output_file):
  methylation_sites = pd.read_csv(cpg_file)
  methylation_sites = list(methylation_sites.iloc[3:, 0])

  combined_data = []
  for file_name in os.listdir(data_dir):
    if file_name.endswith("_series_matrix.txt"):
      file_path = os.path.join(data_dir, file_name)
      
      ages, genders, tissues = extract_metadata(file_path)
      
      methylation_data = filter_methylation_data(file_path, methylation_sites)
      
      for idx, (age, gender, tissue) in enumerate(zip(ages, genders, tissues)):
        sample_data = methylation_data.iloc[:, idx].copy()
        sample_data['Age'] = age
        sample_data['Gender'] = gender
        sample_data['Tissue'] = tissue
        combined_data.append(sample_data)

  final_dataset = pd.concat(combined_data, axis=1).T
  print(final_dataset.head())
  print(final_dataset.shape)
  raise Exception
  final_dataset.to_csv(output_file, index=False)
  print(f"Preprocessed data saved")


if __name__ == "__main__":
  preprocess_data_dir("data/", "CpGs.csv", "preprocessed_dataset.csv")
  
