import argparse
import logging
import pandas as pd
from ours.cross_school.KSCD_high import KSCD
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

# Parameter setting section--------------------------------------------------------
parser = argparse.ArgumentParser(description='Your program description here.')
parser.add_argument('--rate', type=float, default=0.1, help='Description of rate parameter')
parser.add_argument('--pp_dim', type=int, default=20, help='Description of pp_dim parameter')
parser.add_argument('--low_dim', type=int, default=20, help='Description of low_dim parameter')
parser.add_argument('--batch_size', type=int, default=256, help='Description of batch_size parameter')
parser.add_argument('--model_file', type=str, default="source_model/cross_school/kscd/temp.pth", help='')
parser.add_argument('--target_model_file', type=str, default="target_model/cross_school/kscd/temp.pth", help='')
parser.add_argument('--if_source_train', type=int, default=0, help='Description of if_source_train parameter')
parser.add_argument('--if_target_migration', type=int, default=1, help='0 - Origin, 1 - Ours, 2 - ParTran')
parser.add_argument('--folder', type=str, default='data1/B+C+D_A/', help='Folder path for data')
parser.add_argument('--source', type=str, default='schoolB,schoolC,schoolD', help='Names of source schools separated by comma')
parser.add_argument('--target', type=str, default='schoolA', help='Name of target school')


args = parser.parse_args()

rate = args.rate
pp_dim = args.pp_dim
batch_size = args.batch_size
model_file = args.model_file
if_source_train = args.if_source_train
if_target_migration = args.if_target_migration
folder = args.folder
source = args.source
target = args.target
low_dim  = args.low_dim
target_model_file = args.target_model_file
#-----------------------------------------------------------------
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
item = pd.read_csv(f"{folder}/item.csv")
#----------------------------------------------------------------------

Source_df = pd.concat(list(Source_data.values()), ignore_index=True)
Source_train, Source_test = train_test_split(Source_df, test_size=0.2, random_state=42)
Source_train.reset_index(inplace=True, drop=True)
Source_test.reset_index(inplace=True, drop=True)

exer_set = Target["item_id"].drop_duplicates()
rand_stu = exer_set.sample(frac=rate, random_state=2024)
Target_fine_tuning = Target[Target["item_id"].isin(rand_stu)]
Target_test = Target[~Target["item_id"].isin(rand_stu)]
Target_fine_tuning.reset_index(inplace=True, drop=True)
Target_test.reset_index(inplace=True, drop=True)

# Split Target_fine_tuning into 9:1
Target_train, Target_val = train_test_split(Target_fine_tuning, test_size=0.1, random_state=42)
Target_train.reset_index(inplace=True, drop=True)
Target_val.reset_index(inplace=True, drop=True)

# Calculate the item range for each source data
source_ranges = {}
for source_name, source_df in Source_data.items():
    source_ranges[source_name] = [source_df['user_id'].min(), source_df['user_id'].max()]
source_range = list(source_ranges.values())

# Record the mapping between exercises and knowledge points
item2knowledge = {}
# Record the number of knowledge points
knowledge_set = set()
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
    for idx in range(len(item)):
        knowledge_emb[idx][np.array(item2knowledge[item[idx]]) - 1] = 1.0
    # Concatenate the two tensors horizontally
    item2 = item + k * item_n
    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64) - 1,  # (1, user_n) to (0, user_n-1)
        torch.tensor(item2, dtype=torch.int64) - 1, # (1, item_n) to (0, item_n-1)
        knowledge_emb,
        torch.tensor(score, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


def transform3(user, item, item2knowledge, score, batch_size, knowledge_n, pp_dim):
    knowledge_emb = torch.zeros((len(item), knowledge_n))
    for idx in range(len(item)):
        knowledge_emb[idx][np.array(item2knowledge[item[idx]]) - 1] = 1.0

    # Used to store the indices of rows to keep
    indices_to_keep = []

    # Iterate through each row of the tensor
    for i, row in enumerate(knowledge_emb):
        # If all elements in the row are not zero, keep the row
        if not torch.all(row == 0):
            indices_to_keep.append(i)

    # Reconstruct the knowledge_emb tensor based on the kept row indices
    knowledge_emb_filtered = knowledge_emb[indices_to_keep]

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

# Create KSCD object
cdm = KSCD(knowledge_n, item_n, s_user_n, t_user_n, low_dim, pp_dim,
           source_range, model_file, target_model_file)

if if_source_train == 1:
    # Output source domain training start prompt to file
    with open("record.txt", "a") as f:
        f.write(f"------------------------------------------------------\n"
                f"KSCD Source domain training\n"
                f"Source: {source}, Target: {target}\n")
    # Train on the source domain
    cdm.Source_train(s_train_set, s_test_set, device="cpu")
    # Output source domain training end prompt to file
    with open("record.txt", "a") as f:
        f.write("-------------------------------------------------------\n")
    logging.info("Source domain training completed.")

if if_target_migration == 1:
    # Transfer parameters
    cdm.Transfer_parameters(cdm.kscd_t_net, source_range)
    logging.info("Transfer parameters to the target domain completed.")
    # Output target domain training start prompt to file
    with open("record.txt", "a") as f:
        f.write(f"------------------------------------------------------\n"
                f"KSCD--our\n"
                f"Source: {source}, Target: {target}\n")
    # Train on the target domain
    cdm.Target_train(cdm.kscd_t_net, t_train_set, t_val_set, epoch=100, device="cuda")
    logging.info("Target domain training completed.")
    # Test the model
    cdm.kscd_t_net.load_state_dict(torch.load(target_model_file))

    auc, accuracy, rmse, f1 = cdm.Target_test(cdm.kscd_t_net, t_test_set)

    # Output best metrics to file
    with open("record.txt", "a") as f:
        f.write("Best AUC: %.6f, Best Accuracy: %.6f, Best RMSE: %.6f, Best F1: %.6f\n" % (
            auc, accuracy, rmse, f1))
        f.write("-------------------------------------------------------\n")

elif if_target_migration == 2:
    # Transfer parameters
    cdm.Transfer_parameters(cdm.kscd_t_net2, source_range)
    logging.info("Transfer parameters to the target domain completed.")
    # Output target domain training start prompt to file
    with open("record.txt", "a") as f:
        f.write(f"-------------------------------------------------------\n"
                f"KSCD--ours++\n"
                f"Source: {source}, Target: {target}\n")
    # Train on the target domain
    cdm.Target_train(cdm.kscd_t_net2, t_train_set, t_val_set, epoch=100, device="cuda")
    logging.info("Target domain training completed.")
    # Test the model
    cdm.kscd_t_net2.load_state_dict(torch.load(target_model_file))

    auc, accuracy, rmse, f1 = cdm.Target_test(cdm.kscd_t_net2, t_test_set)

    # Output best metrics to file
    with open("record.txt", "a") as f:
        f.write("Best AUC: %.6f, Best Accuracy: %.6f, Best RMSE: %.6f, Best F1: %.6f\n" % (
            auc, accuracy, rmse, f1))
        f.write("-------------------------------------------------------\n")

else:
    # Output target domain training start prompt to file
    with open("record.txt", "a") as f:
        f.write(f"-------------------------------------------------------\n"
                f"KSCD-Origin\n"
                f"Source: {source}, Target: {target}\n")
    # Train on the target domain
    cdm.Target_train(cdm.kscd_t_net, t_train_set, t_val_set, epoch=100, device="cuda")
    logging.info("Target domain training completed.")
    # Test the model
    cdm.kscd_t_net.load_state_dict(torch.load(target_model_file))

    auc, accuracy, rmse, f1 = cdm.Target_test(cdm.kscd_t_net, t_test_set)

    # Output best metrics to file
    with open("record.txt", "a") as f:
        f.write("Best AUC: %.6f, Best Accuracy: %.6f, Best RMSE: %.6f, Best F1: %.6f\n" % (
            auc, accuracy, rmse, f1))
        f.write("-------------------------------------------------------\n")


