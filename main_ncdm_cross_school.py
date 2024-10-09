# coding: utf-8
import logging
from sklearn.model_selection import train_test_split
from ours.cross_school.NCDM import NCDM
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import time
import argparse
#----------------------------------------------------------------------
# Create command-line parser
parser = argparse.ArgumentParser(description='Description of your program')

# Add command-line arguments
parser.add_argument('--rate', type=float, default=0.2, help='The ratio for splitting cold start training exercises')
parser.add_argument('--pp_dim', type=int, default=10, help='The dimension of prompts')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
parser.add_argument('--latent_dim', type=int, default=20, help='Description of latent_dim parameter')
parser.add_argument('--model_file', type=str, default="source_model/cross_school/ncdm/temp.pth", help='')
parser.add_argument('--target_model_file', type=str, default="target_model/cross_school/ncdm/temp.pth", help='')
parser.add_argument('--if_source_train', type=int, default=1, choices=[0, 1], help='Whether to train on the source domain (0 for no, 1 for yes)')
parser.add_argument('--if_target_migration', type=int, default=1, choices=[0, 1, 2], help='(0 for Origin, 1 for ours, 2 for ParTrans)')
parser.add_argument('--folder', type=str, default='data1/B+C+D_A/', help='Folder path for data')
parser.add_argument('--source', type=str, default='schoolB,schoolC,schoolD', help='Names of source schools separated by comma')
parser.add_argument('--target', type=str, default='schoolA', help='Name of target school')

# Parse command-line arguments
args = parser.parse_args()

# Now you can access command-line arguments via the args object
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
#---------------------------------------------------------------------
# The entire model training process consists of two stages:
# 1. Training on multiple source domains
# 2. Fine-tuning and testing on the target domain
# Read stage one data
Source_data = {}
Source_items = {}
for source_name in source.split(','):
    Source_data[source_name] = pd.read_csv(f"{folder}/{source_name}.csv")
# Read stage two data
Target = pd.read_csv(f"{folder}/{target}.csv")

#----------------------------------------------------------------------
Source_df = pd.concat(list(Source_data.values()), ignore_index=True)
Source_train, Source_test = train_test_split(Source_df, test_size=0.2, random_state=42)
Source_train.reset_index(inplace=True, drop=True)
Source_test.reset_index(inplace=True, drop=True)

# Split exercises set
exer_set = Target["user_id"].drop_duplicates()
rand_stu = exer_set.sample(frac=rate, random_state=2024)
Target_fine_tuning = Target[Target["user_id"].isin(rand_stu)]
Target_test = Target[~Target["user_id"].isin(rand_stu)]
Target_fine_tuning.reset_index(inplace=True, drop=True)
Target_test.reset_index(inplace=True, drop=True)

# Split Target_fine_tuning into 9:1
Target_train, Target_val = train_test_split(Target_fine_tuning, test_size=0.1, random_state=42)
Target_train.reset_index(inplace=True, drop=True)
Target_val.reset_index(inplace=True, drop=True)

user_id_list = Target_fine_tuning['user_id'].unique().tolist()

# Calculate the item range for each source data
source_ranges = {}
for source_name, source_df in Source_data.items():
    source_ranges[source_name] = [source_df['user_id'].min(), source_df['user_id'].max()]
source_range = list(source_ranges.values())

item2knowledge = {}  # Record the mapping between exercises and knowledge points
knowledge_set = set()  # Record the number of knowledge points
for i, s in item.iterrows():
    item_id, knowledge_codes = s['item_id'], list(set(eval(s['knowledge_code'])))
    item2knowledge[item_id] = knowledge_codes
    knowledge_set.update(knowledge_codes)

s_user_n = np.max([np.max(Source_train['user_id']), np.max(Source_test['user_id'])])
t_user_n = np.max([np.max(Target_fine_tuning['user_id']), np.max(Target_test['user_id'])])
item_n = np.max(item['item_id'])
knowledge_n = np.max(list(knowledge_set))


def transform(user, item, item2knowledge, score, batch_size, knowledge_n, pp_dim):
    knowledge_emb = torch.zeros((len(item), knowledge_n))
    prompt_emb = torch.ones((len(item), pp_dim))  # -----------------------------------
    for idx in range(len(item)):
        knowledge_emb[idx][np.array(item2knowledge[item[idx]]) - 1] = 1.0
    # Concatenate the two tensors horizontally
    combined_emb = torch.cat((prompt_emb, knowledge_emb), dim=1)

    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64) - 1,  # (1, user_n) to (0, user_n-1)
        torch.tensor(item, dtype=torch.int64) - 1,  # (1, item_n) to (0, item_n-1)
        combined_emb,
        torch.tensor(score, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)

def transform2(user, item, item2knowledge, score, k, batch_size, knowledge_n, pp_dim):
    knowledge_emb = torch.zeros((len(item), knowledge_n))
    prompt_emb = torch.ones((len(item), pp_dim))
    for idx in range(len(item)):
        knowledge_emb[idx][np.array(item2knowledge[item[idx]]) - 1] = 1.0
    # Concatenate the two tensors horizontally
    combined_emb = torch.cat((prompt_emb, knowledge_emb), dim=1)
    item2 = item + k * item_n
    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64) - 1,  # (1, user_n) to (0, user_n-1)
        torch.tensor(item, dtype=torch.int64) - 1,
        torch.tensor(item2, dtype=torch.int64) - 1, # (1, item_n) to (0, item_n-1)
        knowledge_emb,
        torch.tensor(score, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)

def transform3(user, item, item2knowledge, score, batch_size, knowledge_n, pp_dim):
    knowledge_emb = torch.zeros((len(item), knowledge_n))
    for idx in range(len(item)):
        knowledge_emb[idx][np.array(item2knowledge[item[idx]]) - 1] = 1.0

    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64) - 1,  # (1, user_n) to (0, user_n-1)
        torch.tensor(item, dtype=torch.int64) - 1,
        knowledge_emb,
        torch.tensor(score, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)

s_train_set, s_test_set = [
    transform2(data["user_id"], data["item_id"], item2knowledge, data["score"], data["Source_id"], batch_size, knowledge_n, pp_dim)
    for data in [Source_train, Source_test]
]

t_train_set, t_val_set, t_test_set = [
    transform3(data["user_id"], data["item_id"], item2knowledge, data["score"], batch_size, knowledge_n, pp_dim)
    for data in [Target_train, Target_val, Target_test]
]

# Set logging level to INFO
logging.getLogger().setLevel(logging.INFO)

# Create NCDM object
cdm = NCDM(knowledge_n, item_n, s_user_n, t_user_n, pp_dim,
           source_range, model_file, target_model_file)

if if_source_train == 1:
    # Output source domain training start prompt to file
    with open("record.txt", "a") as f:
        f.write(f"------------------------------------------------------\n"
                f"NCDM Source domain training\n"
                f"Source: {source}, Target: {target}\n")
    # Train on the source domain
    cdm.Source_train(s_train_set, s_test_set, device="cuda")
    # Output source domain training end prompt to console and file
    with open("record.txt", "a") as f:
        f.write("-------------------------------------------------------\n")
    logging.info("Source domain training completed. ")

if if_target_migration == 1:
    # Transfer
    cdm.Transfer_parameters(cdm.ncdm_t_net, source_range)
    logging.info("Transfer parameters to the target domain completed.")
    # Output target domain training start prompt to file
    with open("record.txt", "a") as f:
        f.write(f"------------------------------------------------------\n"
                f"NCDM--our\n"
                f"Source: {source}, Target: {target}\n")
    # Train on the target domain
    cdm.Target_train(cdm.ncdm_t_net, t_train_set, t_test_set, epoch=100, device="cuda")
    logging.info("Target domain training completed.")
    # Test the model
    cdm.ncdm_t_net.load_state_dict(torch.load(target_model_file))

    auc, accuracy, rmse, f1 = cdm.Target_test(cdm.ncdm_t_net, t_test_set)

    # Output best metrics to file
    with open("record.txt", "a") as f:
        f.write("Best AUC: %.6f, Best Accuracy: %.6f, Best RMSE: %.6f, Best F1: %.6f\n" % (
            auc, accuracy, rmse, f1))
        f.write("-------------------------------------------------------\n")

    # cdm.draw_student_distribution3_1()

elif if_target_migration == 2:
    # Transfer
    cdm.Transfer_parameters(cdm.ncdm_t_net2, source_range)
    logging.info("Transfer parameters to the target domain completed.")
    # Output target domain training start prompt to file
    with open("record.txt", "a") as f:
        f.write(f"------------------------------------------------------\n"
                f"NCDM-ours++\n"
                f"Source: {source}, Target: {target}\n")
    # Train on the target domain
    cdm.Target_train(cdm.ncdm_t_net2, t_train_set, t_test_set, epoch=100, device="cuda")
    logging.info("Target domain training completed.")
    # Test the model
    cdm.ncdm_t_net2.load_state_dict(torch.load(target_model_file))

    auc, accuracy, rmse, f1 = cdm.Target_test(cdm.ncdm_t_net2, t_test_set)

    # Output best metrics to file
    with open("record.txt", "a") as f:
        f.write("Best AUC: %.6f, Best Accuracy: %.6f, Best RMSE: %.6f, Best F1: %.6f\n" % (
            auc, accuracy, rmse, f1))
        f.write("-------------------------------------------------------\n")
else:
    # Output target domain training start prompt to file
    with open("record.txt", "a") as f:
        f.write(f"------------------------------------------------------\n"
                f"NCDM-Origin\n"
                f"Source: {source}, Target: {target}\n")
    # Train on the target domain
    cdm.Target_train(cdm.ncdm_t_net, t_train_set, t_test_set, epoch=100, device="cuda")
    logging.info("Target domain training completed.")
    # Test the model
    cdm.ncdm_t_net.load_state_dict(torch.load(target_model_file))

    auc, accuracy, rmse, f1 = cdm.Target_test(cdm.ncdm_t_net, t_test_set)

    # Output best metrics to file
    with open("record.txt", "a") as f:
        f.write("Best AUC: %.6f, Best Accuracy: %.6f, Best RMSE: %.6f, Best F1: %.6f\n" % (
            auc, accuracy, rmse, f1))
        f.write("-------------------------------------------------------\n")