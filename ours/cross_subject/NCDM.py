# coding: utf-8
import csv
import logging
from collections import defaultdict
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score
from torch.utils.data import DataLoader, TensorDataset, random_split

class MappingModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MappingModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        x = F.sigmoid(x)
        return x

class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Transform_Stu(nn.Module):
    def __init__(self, stu_num, pp_dim):
        super(Transform_Stu, self).__init__()
        self.ability_emb = nn.Embedding(stu_num, pp_dim)

    def forward(self, x, stu_id):
        ability_x = self.ability_emb(stu_id)
        output = torch.cat([ability_x, x], dim=1)
        return output


class Transform_Exr(nn.Module):
    def __init__(self, pp_dim, s_ranges):
        super(Transform_Exr, self).__init__()
        self.s_exer_vectors = nn.ParameterList([nn.Parameter(torch.rand(pp_dim)) for _ in range(len(s_ranges))])
        self.mlp = SimpleMLP(pp_dim * len(s_ranges), 10, pp_dim)

    def forward(self, x):
        exer_vectors = torch.cat([vector.unsqueeze(0) for vector in self.s_exer_vectors], dim=1)
        new_exer_vector = self.mlp(exer_vectors)
        new_exer_vector = new_exer_vector.expand(x.size(0), -1)
        output = torch.cat([new_exer_vector, x], dim=1)
        return output


class ConvolutionalTransform(nn.Module):
    def __init__(self, fc_out_features, input_channels=2, output_channels=1, kernel_size=1, stride=1, padding=0):
        super(ConvolutionalTransform, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size, stride, padding)
        self.MLP = SimpleMLP(fc_out_features,10,fc_out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.MLP(x)
        return x

class Source_Net(nn.Module):
    def __init__(self, knowledge_n, exer_n, student_n, pp_dim, s_ranges):
        self.pp_dim = pp_dim
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim + self.pp_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable
        self.s_ranges = s_ranges
        super(Source_Net, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Parameter(torch.rand((self.exer_n, self.knowledge_dim)))
        nn.init.xavier_uniform_(self.k_difficulty)

        self.s_exer_vectors = nn.ParameterList([nn.Parameter(torch.rand(pp_dim)) for _ in range(len(s_ranges))])

        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        #--------------------------------------------------
        self.transform_layer_stu = Transform_Stu(self.emb_num, self.pp_dim)

        #--------------------------------------------------
        self.prednet_full1 = PosLinear(self.knowledge_dim, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        self.fc1 = nn.Linear(self.pp_dim + self.knowledge_dim, self.knowledge_dim)
        self.fc2 = nn.Linear(self.pp_dim + self.knowledge_dim, self.knowledge_dim)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        com_sta_emb = self.transform_layer_stu(stu_emb, stu_id)
        com_sta_emb = torch.sigmoid(self.fc1(com_sta_emb))

        temp_vectors = torch.cat([vector.repeat(r[1] - r[0] + 1, 1) for vector, r in zip(self.s_exer_vectors, self.s_ranges)], dim=0)
        new_k_difficulty_emb = torch.cat([temp_vectors, self.k_difficulty], dim=1)

        k_difficulty = torch.index_select(new_k_difficulty_emb, dim=0, index=input_exercise)
        com_k_difficulty = torch.sigmoid(self.fc2(k_difficulty))

        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))  # * 10
        #----------------------------------------------------------------
        # prednet
        input_x = e_difficulty * (com_sta_emb - com_k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)


class Target_Net(nn.Module):
    def __init__(self,train_knowledge_n, knowledge_n, exer_n, student_n,s_ranges,pp_dim):
        self.pp_dim = pp_dim
        self.train_knowledge_dim = train_knowledge_n
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim + self.pp_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable
        self.s_ranges = s_ranges

        super(Target_Net, self).__init__()
        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.knowledge_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        # --------------------------------------------------
        self.ability_emb = nn.Embedding(self.emb_num, pp_dim)
        self.transform_layer_exr = Transform_Exr(self.pp_dim,self.s_ranges)
        # --------------------------------------------------
        self.prednet_full1 = PosLinear(self.knowledge_dim, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        self.fc1 = nn.Linear(self.pp_dim + self.knowledge_dim, self.knowledge_dim)
        self.fc2 = nn.Linear(self.pp_dim + self.knowledge_dim, self.knowledge_dim)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        ability_x = self.ability_emb(stu_id)
        com_sta_emb = torch.cat([ability_x, stu_emb], dim=1)
        com_sta_emb = torch.sigmoid(self.fc1(com_sta_emb))

        k_difficulty = self.k_difficulty(input_exercise)
        com_k_difficulty = self.transform_layer_exr(k_difficulty)
        com_k_difficulty = torch.sigmoid(self.fc2(com_k_difficulty))

        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))  # * 10
        # ----------------------------------------------------------------
        # prednet
        input_x = e_difficulty * (com_sta_emb - com_k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)

class Target_Net2(nn.Module):

    def __init__(self,train_knowledge_n, knowledge_n, exer_n, student_n,s_ranges,pp_dim):
        self.pp_dim = pp_dim
        self.train_knowledge_dim = train_knowledge_n
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim + self.pp_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable
        self.s_ranges = s_ranges

        super(Target_Net2, self).__init__()
        # prediction sub-net
        #self.student_emb = nn.Embedding(self.emb_num, self.knowledge_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        # --------------------------------------------------
        #self.transform_layer_stu = Transform_Stu(self.emb_num, self.pp_dim)
        self.ability_emb = nn.Embedding(self.emb_num, pp_dim)
        self.generalize_layer_stu = nn.Linear(self.pp_dim, self.knowledge_dim)

        self.transform_layer_exr = Transform_Exr(self.pp_dim,self.s_ranges)
        # --------------------------------------------------
        self.prednet_full1 = PosLinear(self.knowledge_dim, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        self.fc1 = nn.Linear(self.pp_dim + self.knowledge_dim, self.knowledge_dim)
        self.fc2 = nn.Linear(self.pp_dim + self.knowledge_dim, self.knowledge_dim)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        #stu_emb = self.student_emb(stu_id)
        ability_x = self.ability_emb(stu_id)
        stu_emb = self.generalize_layer_stu(ability_x)
        com_sta_emb = torch.cat([ability_x, stu_emb], dim=1)
        com_sta_emb = torch.sigmoid(self.fc1(com_sta_emb))

        k_difficulty = self.k_difficulty(input_exercise)
        com_k_difficulty = self.transform_layer_exr(k_difficulty)
        com_k_difficulty = torch.sigmoid(self.fc2(com_k_difficulty))

        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))  # * 10
        # ----------------------------------------------------------------
        # prednet
        input_x = e_difficulty * (com_sta_emb - com_k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)

class Net(nn.Module):

    def __init__(self, knowledge_n, exer_n, student_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))  # * 10
        # prednet
        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)


class NCDM:
    '''Neural Cognitive Diagnosis Model'''

    def __init__(self, source_knowledge_n, target_knowledge_n, source_exer_n, target_exer_n, student_n,pp_dim,s_ranges,model_file, target_model_file):
        super(NCDM, self).__init__()
        self.pp_dim = pp_dim
        self.latent_dim = target_knowledge_n
        self.model_file = model_file
        self.target_model_file = target_model_file
        self.ncdm_s_net = Source_Net(source_knowledge_n, source_exer_n, student_n, pp_dim,s_ranges)
        self.ncdm_t_net = Target_Net(source_knowledge_n, target_knowledge_n, target_exer_n, student_n, s_ranges, pp_dim)
        self.ncdm_t_net2 = Target_Net2(source_knowledge_n, target_knowledge_n, target_exer_n, student_n, s_ranges, pp_dim)

    def Source_train(self, train_data, test_data=None, max_epochs=50, early_stopping_patience=2, device="cpu",
                     lr=0.001, silence=False):
        self.ncdm_s_net = self.ncdm_s_net.to(device)
        self.ncdm_s_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.ncdm_s_net.parameters(), lr=lr)

        epoch = 0
        best_auc = 0.0
        consecutive_no_improvement = 0

        while max_epochs is None or epoch < max_epochs:
            epoch_losses = []
            batch_count = 0

            for batch_data in tqdm(train_data, "Epoch %s" % epoch):
                batch_count += 1
                user_id, item_id, knowledge_emb, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = y.to(device)

                pred: torch.Tensor = self.ncdm_s_net(user_id, item_id, knowledge_emb)
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())
            #print(self.ncdm_s_net.transform_layer_stu.ability_emb.weight)
            average_loss = float(np.mean(epoch_losses))
            print("[Epoch %d] average loss: %.6f" % (epoch, average_loss))

            if test_data is not None:
                auc, accuracy = self.Source_net_eval(test_data, device=device)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (epoch, auc, accuracy))

                e = auc - best_auc
                if e > 0.001:
                    best_auc = auc
                    consecutive_no_improvement = 0

                    torch.save(self.ncdm_s_net.state_dict(), self.model_file)
                    print(f"Saved the best model with AUC: {best_auc} at epoch {epoch}")

                else:
                    if e > 0:
                        best_auc = auc
                    consecutive_no_improvement += 1

                # Early stopping check
                if early_stopping_patience is not None and consecutive_no_improvement >= early_stopping_patience:
                    print(
                        f"Early stopping at epoch {epoch} as there is no improvement in {early_stopping_patience} consecutive epochs.")
                    break

            epoch += 1

        with open("record.txt", "a") as f:
            f.write(f"Best AUC: {best_auc}, Epoch: {epoch}\n")

    def Target_train(self, model, train_data, test_data=None, epoch=50, device="cpu", lr=0.002, silence=False, patience=2):
        ncdm_t_net = model.to(device)
        ncdm_t_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(ncdm_t_net.parameters(), lr=lr)

        best_auc = 0.0
        best_metrics = None
        early_stop_counter = 0

        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                user_id, item_id, knowledge_emb, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = y.to(device)
                pred: torch.Tensor = ncdm_t_net(user_id, item_id, knowledge_emb)
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            average_loss = np.mean(epoch_losses)
            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(average_loss)))

            if test_data is not None:
                auc, accuracy, rmse, f1 = self.Target_net_eval(model, test_data, device=device)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (epoch_i, auc, accuracy))

                e = auc - best_auc
                # Update best metrics if current metrics are better
                if e > 0.0001:
                    best_auc = auc
                    best_metrics = (auc, accuracy, rmse, f1)
                    early_stop_counter = 0  # 重置早停计数器
                    torch.save(ncdm_t_net.state_dict(), self.target_model_file)
                    print(f"Saved the best target model with AUC: {best_auc} at epoch {epoch}")
                else:
                    if e > 0:
                        best_auc = auc
                        best_metrics = (auc, accuracy, rmse, f1)
                    early_stop_counter += 1

                # Check for early stopping
                if early_stop_counter >= patience:
                    print(f"Early stopping at epoch {epoch_i}. No improvement for {patience} epochs.")
                    break

    def Target_train_0(self, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False, patience=3):
        self.ncdm_t_net = self.ncdm_t_net.to(device)
        self.ncdm_t_net.train()

        best_auc = 0.0
        best_metrics = None
        early_stop_counter = 0

        for epoch_i in range(epoch):
            if test_data is not None:
                auc, accuracy, rmse, f1 = self.Target_net_eval(test_data, device=device)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (epoch_i, auc, accuracy))

                # Update best metrics if current metrics are better
                if auc > best_auc:
                    best_auc = auc
                    best_metrics = (auc, accuracy, rmse, f1)
                    early_stop_counter = 0  # 重置早停计数器
                else:
                    early_stop_counter += 1

                # Check for early stopping
                if early_stop_counter >= patience:
                    print(f"Early stopping at epoch {epoch_i}. No improvement for {patience} epochs.")
                    break

        if best_metrics is not None:
            best_auc, best_accuracy, best_rmse, best_f1 = best_metrics
            print("Best AUC: %.6f, Best Accuracy: %.6f, Best RMSE: %.6f, Best F1: %.6f" % (
                best_auc, best_accuracy, best_rmse, best_f1))

    def Source_net_eval(self, test_data, device="cpu"):
        self.ncdm_s_net = self.ncdm_s_net.to(device)
        self.ncdm_s_net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred: torch.Tensor = self.ncdm_s_net(user_id, item_id, knowledge_emb)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def Target_net_eval(self, model, test_data, device="cpu"):
        ncdm_t_net = model.to(device)
        ncdm_t_net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred: torch.Tensor = ncdm_t_net(user_id, item_id, knowledge_emb)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        y_pred_binary = np.array(y_pred) >= 0.5
        f1 = f1_score(y_true, y_pred_binary)

        auc = roc_auc_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred_binary)

        return auc, accuracy, rmse, f1


    def Transfer_parameters(self, model, s_ranges):
        self.ncdm_s_net.load_state_dict(torch.load(self.model_file))
        model.ability_emb.weight.data.copy_(
            self.ncdm_s_net.transform_layer_stu.ability_emb.weight.data)

        for i in range(len(s_ranges)):
            model.transform_layer_exr.s_exer_vectors[i].data.copy_(
                self.ncdm_s_net.s_exer_vectors[i].data)
            model.transform_layer_exr.s_exer_vectors[i].requires_grad = True

        model.prednet_full2.weight.data = self.ncdm_s_net.prednet_full2.weight.data.clone()
        model.prednet_full2.bias.data = self.ncdm_s_net.prednet_full2.bias.data.clone()

        model.prednet_full3.weight.data = self.ncdm_s_net.prednet_full3.weight.data.clone()
        model.prednet_full3.bias.data = self.ncdm_s_net.prednet_full3.bias.data.clone()


    def Target_test(self, model, test_data, device="cpu"):
        ncdm_t_net = model.to(device)
        ncdm_t_net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred: torch.Tensor = ncdm_t_net(user_id, item_id, knowledge_emb)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        y_pred_binary = np.array(y_pred) >= 0.5
        f1 = f1_score(y_true, y_pred_binary)

        auc = roc_auc_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred_binary)

        return auc, accuracy, rmse, f1