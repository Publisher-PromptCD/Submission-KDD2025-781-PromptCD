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
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--rate', type=float, default=0.2, help='The ratio for splitting cold start training exercises')
parser.add_argument('--pp_dim', type=int, default=10, help='The dimension of prompts')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
parser.add_argument('--model_file', type=str, default="source_model/cross_school/ncdm/temp.pth", help='')
parser.add_argument('--target_model_file', type=str, default="target_model/cross_school/ncdm/temp.pth", help='')
parser.add_argument('--if_source_train', type=int, default=1, choices=[0, 1], help='Whether to train on the source domain (0 for no, 1 for yes)')
parser.add_argument('--if_target_migration', type=int, default=1, choices=[0, 1, 2], help='(0 for Origin, 1 for ours, 2 for ParTrans)')
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
target_model_file = args.target_model_file
#-----------------------------------------------------------------
#整个模型训练过程分为两个阶段：1、在多个源域上进行训练  2、在目标域上进行微调、测试
# 读取阶段一数据
Source_data = {}
Source_items = {}
for source_name in source.split(','):
    Source_data[source_name] = pd.read_csv(f"{folder}/{source_name}.csv")

# 读取阶段二数据
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

#对 Target_fine_tuning 进行 9:1 划分
Target_train, Target_val = train_test_split(Target_fine_tuning, test_size=0.1, random_state=42)
Target_train.reset_index(inplace=True, drop=True)
Target_val.reset_index(inplace=True, drop=True)

# 计算每个源数据的项目范围
source_ranges = {}
for source_name, source_df in Source_data.items():
    source_ranges[source_name] = [source_df['user_id'].min(), source_df['user_id'].max()]
source_range = list(source_ranges.values())

item2knowledge = {}  # 记录练习题和知识点的映射关系
knowledge_set = set()  # 记录知识点数量
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
    # 在水平方向拼接两个张量
    combined_emb = torch.cat((prompt_emb, knowledge_emb), dim=1)

    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64) - 1,  # (1, user_n) to (0, user_n-1)
        torch.tensor(item, dtype=torch.int64) - 1,  # (1, item_n) to (0, item_n-1)
        combined_emb,
        torch.tensor(score, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)

def transform2(user, item, item2knowledge, score, k, batch_size,knowledge_n,pp_dim):
    knowledge_emb = torch.zeros((len(item), knowledge_n))
    prompt_emb = torch.ones((len(item), pp_dim))
    for idx in range(len(item)):
        knowledge_emb[idx][np.array(item2knowledge[item[idx]]) - 1] = 1.0
    # 在水平方向拼接两个张量
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

def transform3(user, item, item2knowledge, score,batch_size,knowledge_n,pp_dim):
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

# 设置日志级别为INFO
logging.getLogger().setLevel(logging.INFO)

# 创建NCDM对象
cdm = NCDM(knowledge_n, item_n, s_user_n, t_user_n, pp_dim,
           source_range, model_file, target_model_file)

if if_source_train == 1:
    # 将源域训练开始提示输出到文件
    with open("record.txt", "a") as f:
        f.write(f"------------------------------------------------------\n"
                f"NCDM Source domain training\n"
                f"Source: {source}, Target: {target}\n")
    # 在源域上训练
    cdm.Source_train(s_train_set, s_test_set, device="cuda")
    # 将源域训练训练结束提示输出到控制台和文件
    with open("record.txt", "a") as f:
        f.write("-------------------------------------------------------\n")
    logging.info("Source domain training completed. ")

if if_target_migration == 1:
    # 迁移
    cdm.Transfer_parameters(cdm.ncdm_t_net,source_range)
    logging.info("Transfer parameters to the target domain completed.")
    # 将目标域训练开始提示输出到文件
    with open("record.txt", "a") as f:
        f.write(f"------------------------------------------------------\n"
                f"NCDM--our\n"
                f"Source: {source}, Target: {target}\n")
    # 目标域训练
    cdm.Target_train(cdm.ncdm_t_net, t_train_set, t_test_set, epoch=100, device="cuda")
    logging.info("Target domain training completed.")
    # 测试
    cdm.ncdm_t_net.load_state_dict(torch.load(target_model_file))

    auc, accuracy, rmse, f1 = cdm.Target_test(cdm.ncdm_t_net, t_test_set)

    # 将最佳指标输出到文件
    with open("record.txt", "a") as f:
        f.write("Best AUC: %.6f, Best Accuracy: %.6f, Best RMSE: %.6f, Best F1: %.6f\n" % (
            auc, accuracy, rmse, f1))
        f.write("-------------------------------------------------------\n")

elif if_target_migration == 2:
    # 迁移
    cdm.Transfer_parameters(cdm.ncdm_t_net2,source_range)
    logging.info("Transfer parameters to the target domain completed.")
    # 将目标域训练开始提示输出到文件
    with open("record.txt", "a") as f:
        f.write(f"-------------------------------------------------------\n"
                f"NCDM-ours++\n"
                f"Source: {source}, Target: {target}\n")
    # 目标域训练
    cdm.Target_train(cdm.ncdm_t_net2, t_train_set, t_test_set, epoch=100, device="cuda")
    logging.info("Target domain training completed.")
    # 测试
    cdm.ncdm_t_net2.load_state_dict(torch.load(target_model_file))

    auc, accuracy, rmse, f1 = cdm.Target_test(cdm.ncdm_t_net2, t_test_set)

    # 将最佳指标输出到文件
    with open("record.txt", "a") as f:
        f.write("Best AUC: %.6f, Best Accuracy: %.6f, Best RMSE: %.6f, Best F1: %.6f\n" % (
            auc, accuracy, rmse, f1))
        f.write("-------------------------------------------------------\n")
else:
    # 将目标域训练开始提示输出到文件
    with open("record.txt", "a") as f:
        f.write(f"-------------------------------------------------------\n"
                f"NCDM-Origin\n"
                f"Source: {source}, Target: {target}\n")
    # 目标域训练
    cdm.Target_train(cdm.ncdm_t_net, t_train_set, t_test_set, epoch=100, device="cuda")
    logging.info("Target domain training completed.")
    # 测试
    cdm.ncdm_t_net.load_state_dict(torch.load(target_model_file))

    auc, accuracy, rmse, f1 = cdm.Target_test(cdm.ncdm_t_net, t_test_set)

    # 将最佳指标输出到文件
    with open("record.txt", "a") as f:
        f.write("Best AUC: %.6f, Best Accuracy: %.6f, Best RMSE: %.6f, Best F1: %.6f\n" % (
            auc, accuracy, rmse, f1))
        f.write("-------------------------------------------------------\n")