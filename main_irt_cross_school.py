# coding: utf-8
import logging
import time
from sklearn.model_selection import train_test_split
from ours.cross_school.IRT import IRT
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import argparse
#----------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--rate', type=float, default=0.5, help='The ratio for splitting cold start training exercises')
parser.add_argument('--pp_dim', type=int, default=20, help='The dimension of prompts')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
parser.add_argument('--latent_dim', type=int, default=1, help='Description of latent_dim parameter')
parser.add_argument('--model_file', type=str, default="source_model/cross_school/irt/temp.pth", help='')
parser.add_argument('--target_model_file', type=str, default="target_model/cross_school/irt/temp.pth", help='')
parser.add_argument('--if_source_train', type=int, default=0, choices=[0, 1], help='Whether to train on the source domain (0 for no, 1 for yes)')
parser.add_argument('--if_target_migration', type=int, default=2, choices=[0, 1, 2], help='(0 for Origin, 1 for ours, 2 for ParTrans)')
parser.add_argument('--folder', type=str, default='data1/B+C+D_A/', help='Folder path for data')
parser.add_argument('--source', type=str, default='schoolB,schoolC,schoolD', help='Names of source schools separated by comma')
parser.add_argument('--target', type=str, default='schoolA', help='Name of target school')

# Parse command line arguments
args = parser.parse_args()

# You can now access the command line parameters through the args object
rate = args.rate
pp_dim = args.pp_dim
batch_size = args.batch_size
latent_dim = args.latent_dim
model_file = args.model_file
if_source_train = args.if_source_train
if_target_migration = args.if_target_migration
folder = args.folder
source = args.source
target = args.target
target_model_file = args.target_model_file
#----------------------------------------------------------------------

# The entire model training process is divided into two stages:
# 1. Train on multiple source domains
# 2. Fine-tune and test on the target domain
# Load data for stage one
Source_data = {}
for source_name in source.split(','):
    Source_data[source_name] = pd.read_csv(f"{folder}/{source_name}.csv")
# Load data for stage two
Target = pd.read_csv(f"{folder}/{target}.csv")
item = pd.read_csv(f"{folder}/item.csv")
#-----------------------------------------------------------------------

Source_df = pd.concat(list(Source_data.values()), ignore_index=True)
Source_train, Source_test = train_test_split(Source_df, test_size=0.2, random_state=42)
Source_train.reset_index(inplace=True, drop=True)
Source_test.reset_index(inplace=True, drop=True)

# Select a subset of exercises for fine-tuning based on the specified rate
exer_set = Target["item_id"].drop_duplicates()
rand_stu = exer_set.sample(frac=rate, random_state=2024)
Target_fine_tuning = Target[Target["item_id"].isin(rand_stu)]
Target_test = Target[~Target["item_id"].isin(rand_stu)]
Target_fine_tuning.reset_index(inplace=True, drop=True)
Target_test.reset_index(inplace=True, drop=True)

# Split Target_fine_tuning into 90% training and 10% validation
Target_train, Target_val = train_test_split(Target_fine_tuning, test_size=0.1, random_state=42)
Target_train.reset_index(inplace=True, drop=True)
Target_val.reset_index(inplace=True, drop=True)

# Calculate the item range for each source domain
source_ranges = {}
for source_name, source_df in Source_data.items():
    source_ranges[source_name] = [source_df['user_id'].min(), source_df['user_id'].max()]
source_range = list(source_ranges.values())

s_user_n = np.max([np.max(Source_train['user_id']), np.max(Source_test['user_id'])])
t_user_n = np.max([np.max(Target_fine_tuning['user_id']), np.max(Target_test['user_id'])])
item_n = np.max(item['item_id'])


def transform1(x, y, z, k, batch_size, **params):
    # Transformation logic: replace the original x with x + k * x.max
    y_transformed = y + k * item_n

    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.int64) - 1,
        torch.tensor(y, dtype=torch.int64) - 1,
        torch.tensor(y_transformed, dtype=torch.int64) - 1,
        torch.tensor(z, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def transform2(x, y, z, batch_size, **params):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.int64) - 1,
        torch.tensor(y, dtype=torch.int64) - 1,
        torch.tensor(z, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

s_train_set, s_test_set = [
    transform1(data["user_id"], data["item_id"], data["score"], data["Source_id"], batch_size)
    for data in [Source_train, Source_test]
]

t_train_set, t_val_set, t_test_set = [
    transform2(data["user_id"], data["item_id"], data["score"], batch_size)
    for data in [Target_train, Target_val, Target_test]
]

logging.getLogger().setLevel(logging.INFO)

cdm = IRT(s_user_n, t_user_n, item_n, latent_dim,
           pp_dim, source_range, model_file, target_model_file)

if if_source_train == 1:
    # Output source domain training start message to file
    with open("record.txt", "a") as f:
        f.write(f"------------------------------------------------------\n"
                f"IRT Source domain training\n"
                f"Source: {source}, Target: {target}\n")
    # Train on source domain
    cdm.Source_train(s_train_set, s_test_set, device="cuda")
    # Output source domain training completed message to console and file
    with open("record.txt", "a") as f:
        f.write("-------------------------------------------------------\n")
    logging.info("Source domain training completed. ")

if if_target_migration == 1:
    # Transfer parameters
    cdm.Transfer_parameters(cdm.t_irt_net, source_range)
    logging.info("Transfer parameters to the target domain completed.")
    # Output target domain training start message to file
    with open("record.txt", "a") as f:
        f.write(f"------------------------------------------------------\n"
                f"IRT--our\n"
                f"Source: {source}, Target: {target}\n")
    # Train on target domain
    cdm.Target_train(cdm.t_irt_net, t_train_set, t_val_set, epoch=100, device="cuda")
    logging.info("Target domain training completed.")
    # Test the model
    cdm.t_irt_net.load_state_dict(torch.load(target_model_file))

    auc, accuracy, rmse, f1 = cdm.Target_test(cdm.t_irt_net, t_test_set)

    # Output the best metrics to file
    with open("record.txt", "a") as f:
        f.write("Best AUC: %.6f, Best Accuracy: %.6f, Best RMSE: %.6f, Best F1: %.6f\n" % (
            auc, accuracy, rmse, f1))
        f.write("-------------------------------------------------------\n")

elif if_target_migration == 2:
    # Transfer parameters
    cdm.Transfer_parameters(cdm.t_irt_net2, source_range)
    logging.info("Transfer parameters to the target domain completed.")
    # Output target domain training start message to file
    with open("record.txt", "a") as f:
        f.write(f"-------------------------------------------------------\n"
                f"IRT--our++\n"
                f"Source: {source}, Target: {target}\n")
    # Train on target domain
    cdm.Target_train(cdm.t_irt_net2, t_train_set, t_val_set, epoch=100, device="cuda")
    logging.info("Target domain training completed.")
    # Test the model
    cdm.t_irt_net2.load_state_dict(torch.load(target_model_file))

    auc, accuracy, rmse, f1 = cdm.Target_test(cdm.t_irt_net2, t_test_set)

    # Output the best metrics to file
    with open("record.txt", "a") as f:
        f.write("Best AUC: %.6f, Best Accuracy: %.6f, Best RMSE: %.6f, Best F1: %.6f\n" % (
            auc, accuracy, rmse, f1))
        f.write("-------------------------------------------------------\n")
else:
    # Output target domain training start message to file
    with open("record.txt", "a") as f:
        f.write(f"-------------------------------------------------------\n"
                f"IRT-Origin\n"
                f"Source: {source}, Target: {target}\n")
    # Train on target domain
    cdm.Target_train(cdm.t_irt_net, t_train_set, t_val_set, epoch=100, device="cuda")
    logging.info("Target domain training completed.")
    # Test the model
    cdm.t_irt_net.load_state_dict(torch.load(target_model_file))

    auc, accuracy, rmse, f1 = cdm.Target_test(cdm.t_irt_net, t_test_set)

    # Output the best metrics to file
    with open("record.txt", "a") as f:
        f.write("Best AUC: %.6f, Best Accuracy: %.6f, Best RMSE: %.6f, Best F1: %.6f\n" % (
            auc, accuracy, rmse, f1))
        f.write("-------------------------------------------------------\n")
