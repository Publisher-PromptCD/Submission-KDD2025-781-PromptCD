# coding: utf-8
import csv
import logging
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score
from sklearn.preprocessing import MinMaxScaler
import random

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

class Transform_Exr(nn.Module):
    def __init__(self, exer_num, pp_dim):
        super(Transform_Exr, self).__init__()

        self.overall_diff_emb = nn.Embedding(exer_num, pp_dim)

    def forward(self, x, exer_id):
        # Add the vector to the input data
        overall_diff_x = self.overall_diff_emb(exer_id)
        output = torch.cat([overall_diff_x, x], dim=1)
        return output

class Target_Transform_stu(nn.Module):
    def __init__(self, pp_dim, s_ranges):
        super(Target_Transform_stu, self).__init__()
        self.s_stu_vectors = nn.ParameterList([nn.Parameter(torch.rand(pp_dim)) for _ in range(len(s_ranges))])
        # Vertically concatenate through 1D convolution
        self.Conv = ConvolutionalTransform(pp_dim, input_channels=len(s_ranges))
        # Horizontally concatenate through MLP
        # self.mlp = SimpleMLP(pp_dim * len(s_ranges), 10, pp_dim)

    def forward(self, x):
        # Add vectors to input data
        # Vertical concatenation
        stu_vectors = torch.cat([vector.unsqueeze(0) for vector in self.s_stu_vectors], dim=0)
        new_stu_vector = self.Conv(stu_vectors)
        # Horizontal concatenation
        # stu_vectors = torch.cat([vector.unsqueeze(0) for vector in self.s_stu_vectors], dim=1)
        # new_stu_vector = self.mlp(stu_vectors)

        output = torch.cat([new_stu_vector.expand(x.size(0), -1), x], dim=1)
        return output

class ConvolutionalTransform(nn.Module):
    def __init__(self, fc_out_features, input_channels=3, output_channels=1, kernel_size=1, stride=1, padding=0):
        super(ConvolutionalTransform, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size, stride, padding)
        # self.MLP = SimpleMLP(fc_out_features, 10, fc_out_features)
        # self.fc = nn.Linear(fc_out_features, fc_out_features)

    def forward(self, x):
        x = self.conv1(x)
        # x = F.relu(x)
        # Flatten the output to a one-dimensional tensor for input into the fully connected layer
        x = x.view(x.size(0), -1)  # -1 indicates automatic size inference
        # x = self.MLP(x)
        # x = self.fc(x)
        return x

class Source_Net(nn.Module):
    # knowledge_n is used to train the sum of knowledge points in the subject, and exer_n and student_n are calculated based on total numbers
    def __init__(self, knowledge_n, exer_n, student_n, pp_dim, s_ranges):
        super(Source_Net, self).__init__()
        self.pp_dim = pp_dim
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim + self.pp_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable
        self.s_ranges = s_ranges

        # Prediction sub-net
        self.student_emb = nn.Parameter(torch.rand((self.emb_num, self.stu_dim)))
        nn.init.xavier_uniform_(self.student_emb)
        self.s_stu_vectors = nn.ParameterList([nn.Parameter(torch.rand(pp_dim)) for _ in range(len(s_ranges))])

        self.k_difficulty = nn.ParameterList([nn.Parameter(torch.randn(self.exer_n, self.knowledge_dim))
                                               for _ in range(len(s_ranges))])
        # Perform Xavier uniform initialization for each parameter
        for k in self.k_difficulty:
            nn.init.xavier_uniform_(k)
        self.prompt_k_difficulty = nn.Parameter(torch.randn(self.exer_n, self.pp_dim))
        nn.init.xavier_uniform_(self.prompt_k_difficulty)

        self.e_difficulty = nn.ParameterList([nn.Parameter(torch.randn(self.exer_n, 1))
                                               for _ in range(len(s_ranges))])
        # Perform Xavier uniform initialization for each parameter
        for e in self.e_difficulty:
            nn.init.xavier_uniform_(e)
        self.prompt_e_difficulty = nn.Parameter(torch.randn(self.exer_n, 1))
        nn.init.xavier_uniform_(self.prompt_e_difficulty)
        # self.e_difficulty = nn.Embedding(self.exer_n, 1)
        # --------------------------------------------------
        self.prednet_full1 = PosLinear(self.knowledge_dim, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        self.fc1 = nn.Linear(self.prednet_input_len, self.knowledge_dim)
        self.fc2 = nn.Linear(self.prednet_input_len, self.knowledge_dim)
        self.fc3 = nn.Linear(1 + 1, 1)

        # Initialize weights
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_exercise2, input_knowledge_point):
        # Repeat the prompt_e_difficulty parameter n times
        prompt_e_repeated = self.prompt_e_difficulty.repeat(len(self.s_ranges), 1)
        # Vertically concatenate each element in the list
        e_concatenated = torch.cat([e for e in self.e_difficulty], dim=0)
        # Horizontally concatenate two tensors
        new_e_difficulty = torch.cat([prompt_e_repeated, e_concatenated], dim=1)
        new_e_difficulty = torch.index_select(new_e_difficulty, dim=0, index=input_exercise2)
        new_e_difficulty = torch.sigmoid(self.fc3(new_e_difficulty))

        temp_vectors = torch.cat(
            [vector.repeat(r[1] - r[0] + 1, 1) for vector, r in zip(self.s_stu_vectors, self.s_ranges)], dim=0)
        new_stu_emb = torch.cat([temp_vectors, self.student_emb], dim=1)
        sta_emb = torch.index_select(new_stu_emb, dim=0, index=stu_id)
        com_sta_emb = torch.sigmoid(self.fc1(sta_emb))
        # ----------------------------------------------------------------
        # Repeat the prompt_k_difficulty parameter n times
        prompt_k_repeated = self.prompt_k_difficulty.repeat(len(self.s_ranges), 1)
        # Vertically concatenate each element in the list
        k_concatenated = torch.cat([k for k in self.k_difficulty], dim=0)
        # Horizontally concatenate two tensors
        new_k_difficulty = torch.cat([prompt_k_repeated, k_concatenated], dim=1)
        new_k_difficulty = torch.index_select(new_k_difficulty, dim=0, index=input_exercise2)
        com_k_difficulty = torch.sigmoid(self.fc2(new_k_difficulty))

        # ----------------------------------------------------------------
        # Prediction network
        input_x = new_e_difficulty * (com_sta_emb - com_k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)

class Target_Net(nn.Module):
    # train_knowledge_n is the maximum number of knowledge points during training, knowledge_n is the current subject's maximum knowledge points
    # exer_n is the number of exercises in the target domain, student remains the same as before
    def __init__(self, train_knowledge_n, knowledge_n, exer_n, student_n, s_ranges, pp_dim):
        super(Target_Net, self).__init__()
        self.pp_dim = pp_dim
        self.train_knowledge_dim = train_knowledge_n
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim + self.pp_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable
        self.s_ranges = s_ranges

        # Prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.knowledge_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        # --------------------------------------------------
        self.transform_layer_stu = Target_Transform_stu(self.pp_dim, self.s_ranges)
        self.prompt_e_difficulty = nn.Embedding(self.exer_n, 1)
        self.prompt_k_difficulty = nn.Embedding(self.exer_n, self.pp_dim)
        # --------------------------------------------------
        self.prednet_full1 = PosLinear(self.knowledge_dim, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)
        #
        self.fc1 = nn.Linear(self.prednet_input_len, self.knowledge_dim)
        self.fc2 = nn.Linear(self.prednet_input_len, self.knowledge_dim)
        self.fc3 = nn.Linear(1 + 1, 1)

        # Initialize weights
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # Before prediction network
        stu_emb = self.student_emb(stu_id)
        k_difficulty = self.k_difficulty(input_exercise)
        e_difficulty = self.e_difficulty(input_exercise)  # * 10
        # ----------------------------------------------------------------
        com_sta_emb = self.transform_layer_stu(stu_emb)
        com_sta_emb = torch.sigmoid(self.fc1(com_sta_emb))

        p_k = self.prompt_k_difficulty(input_exercise)
        new_k_difficulty = torch.cat([p_k, k_difficulty], dim=1)
        com_k_difficulty = torch.sigmoid(self.fc2(new_k_difficulty))

        p_e = self.prompt_e_difficulty(input_exercise)
        new_e_difficulty = torch.cat([p_e, e_difficulty], dim=1)
        new_e_difficulty = torch.sigmoid(self.fc3(new_e_difficulty))
        # ----------------------------------------------------------------
        # Prediction network
        input_x = new_e_difficulty * (com_sta_emb - com_k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)

class Target_Net2(nn.Module):

    def __init__(self, train_knowledge_n, knowledge_n, exer_n, student_n, s_ranges, pp_dim):
        super(Target_Net2, self).__init__()
        self.pp_dim = pp_dim
        self.train_knowledge_dim = train_knowledge_n
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim + self.pp_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable
        self.s_ranges = s_ranges

        # Prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.knowledge_dim)
        # self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.generalize_layer_k = nn.Linear(self.pp_dim, self.knowledge_dim)
        # self.e_difficulty = nn.Embedding(self.exer_n, 1)
        self.generalize_layer_e = nn.Linear(1, 1)
        # --------------------------------------------------
        self.transform_layer_stu = Target_Transform_stu(self.pp_dim, self.s_ranges)
        self.prompt_e_difficulty = nn.Embedding(self.exer_n, 1)
        self.prompt_k_difficulty = nn.Embedding(self.exer_n, self.pp_dim)
        # --------------------------------------------------
        self.prednet_full1 = PosLinear(self.knowledge_dim, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)
        #
        self.fc1 = nn.Linear(self.prednet_input_len, self.knowledge_dim)
        self.fc2 = nn.Linear(self.prednet_input_len, self.knowledge_dim)
        self.fc3 = nn.Linear(1 + 1, 1)

        # Initialize weights
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # Before prediction network
        stu_emb = self.student_emb(stu_id)
        # k_difficulty = self.k_difficulty(input_exercise)
        # e_difficulty = self.e_difficulty(input_exercise)  # * 10
        # ----------------------------------------------------------------
        com_sta_emb = self.transform_layer_stu(stu_emb)
        com_sta_emb = torch.sigmoid(self.fc1(com_sta_emb))

        p_k = self.prompt_k_difficulty(input_exercise)
        k_difficulty = self.generalize_layer_k(p_k)
        new_k_difficulty = torch.cat([p_k, k_difficulty], dim=1)
        com_k_difficulty = torch.sigmoid(self.fc2(new_k_difficulty))

        p_e = self.prompt_e_difficulty(input_exercise)
        e_difficulty = self.generalize_layer_e(p_e)
        new_e_difficulty = torch.cat([p_e, e_difficulty], dim=1)
        new_e_difficulty = torch.sigmoid(self.fc3(new_e_difficulty))
        # ----------------------------------------------------------------
        # Prediction network
        input_x = new_e_difficulty * (com_sta_emb - com_k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)

class Net(nn.Module):

    def __init__(self, knowledge_n, exer_n, student_n):
        super(Net, self).__init__()
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        # Prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # Initialize weights
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # Before prediction network
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))  # * 10
        # Prediction network
        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)


class NCDM:
    '''Neural Cognitive Diagnosis Model'''

    def __init__(self, knowledge_n, exer_n, s_stu_n, t_stu_n, pp_dim, s_ranges, model_file, target_model_file):
        super(NCDM, self).__init__()
        self.model_file = model_file
        self.target_model_file = target_model_file
        self.ncdm_s_net = Source_Net(knowledge_n, exer_n, s_stu_n, pp_dim, s_ranges)
        self.ncdm_t_net = Target_Net(knowledge_n, knowledge_n, exer_n, t_stu_n, s_ranges, pp_dim)
        self.ncdm_t_net2 = Target_Net2(knowledge_n, knowledge_n, exer_n, t_stu_n, s_ranges, pp_dim)

    def Source_train(self, train_data, test_data=None, max_epochs=50, early_stopping_patience=5, device="cpu",
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

            for batch_data in tqdm(train_data, f"Epoch {epoch}"):
                batch_count += 1
                user_id, item_id, item_id2, knowledge_emb, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                item_id2: torch.Tensor = item_id2.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = y.to(device)

                pred: torch.Tensor = self.ncdm_s_net(user_id, item_id, item_id2, knowledge_emb)
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())
            # print(self.ncdm_s_net.transform_layer_stu.ability_emb.weight)
            average_loss = float(np.mean(epoch_losses))
            print("[Epoch %d] average loss: %.6f" % (epoch, average_loss))

            if test_data is not None:
                auc, accuracy = self.Source_net_eval(self.ncdm_s_net, test_data, device=device)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (epoch, auc, accuracy))

                e = auc - best_auc
                # Save the best model
                if e > 0.001:
                    best_auc = auc
                    consecutive_no_improvement = 0

                    # Save the model
                    torch.save(self.ncdm_s_net.state_dict(), self.model_file)
                    print(f"Saved the best model with AUC: {best_auc} at epoch {epoch}")

                else:
                    if e > 0:
                        best_auc = auc
                    consecutive_no_improvement += 1

                # Early stopping check
                if early_stopping_patience is not None and consecutive_no_improvement >= early_stopping_patience:
                    print(
                        f"Early stopping at epoch {epoch} as there is no improvement in {early_stopping_patience} consecutive epochs."
                    )
                    break

            epoch += 1

        # Output the best metric to a file
        with open("record.txt", "a") as f:
            f.write(f"Best AUC: {best_auc}, Epoch: {epoch}\n")

    def Target_train(self, model, train_data, test_data=None, epoch=50, device="cpu", lr=0.001, silence=False, patience=5):
        ncdm_t_net = model.to(device)
        ncdm_t_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(ncdm_t_net.parameters(), lr=lr)

        best_auc = 0.0  # Initialize to a low value
        best_metrics = None  # Initialize to None
        early_stop_counter = 0  # Early stopping counter

        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, f"Epoch {epoch_i}"):
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
                auc, accuracy, rmse, f1 = self.Target_net_eval(ncdm_t_net, test_data, device=device)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f, RMSE: %.6f, F1: %.6f" % (epoch_i, auc, accuracy, rmse, f1))

                e = auc - best_auc
                # Update best metrics if current metrics are better
                if e > 0.0001:
                    best_auc = auc
                    best_metrics = (auc, accuracy, rmse, f1)
                    early_stop_counter = 0  # Reset early stopping counter
                    torch.save(ncdm_t_net.state_dict(), self.target_model_file)
                    print(f"Saved the best target model with AUC: {best_auc} at epoch {epoch_i}")
                else:
                    if e > 0:
                        best_auc = auc
                        best_metrics = (auc, accuracy, rmse, f1)
                    early_stop_counter += 1

                # Check for early stopping
                if early_stop_counter >= patience:
                    print(f"Early stopping at epoch {epoch_i}. No improvement for {patience} epochs.")
                    break

    def Source_net_eval(self, model, test_data, device="cpu"):
        ncdm_s_net = model.to(device)
        ncdm_s_net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, item_id2, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            item_id2: torch.Tensor = item_id2.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred: torch.Tensor = ncdm_s_net(user_id, item_id, item_id2, knowledge_emb)
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

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # Convert probability values to binary labels (0 or 1) to compute F1 score
        y_pred_binary = np.array(y_pred) >= 0.5
        f1 = f1_score(y_true, y_pred_binary)

        # Calculate AUC and accuracy
        auc = roc_auc_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred_binary)

        return auc, accuracy, rmse, f1

    def Transfer_parameters(self, model, s_ranges):
        self.ncdm_s_net.load_state_dict(torch.load(self.model_file))
        # Load model and transfer parameters
        model.prompt_k_difficulty.weight.data.copy_(
            self.ncdm_s_net.prompt_k_difficulty.data)
        model.prompt_k_difficulty.requires_grad = True

        model.prompt_e_difficulty.weight.data.copy_(
            self.ncdm_s_net.prompt_e_difficulty.data)
        model.prompt_e_difficulty.requires_grad = True

        for i in range(len(s_ranges)):
            model.transform_layer_stu.s_stu_vectors[i].data.copy_(
                self.ncdm_s_net.s_stu_vectors[i].data)
            model.transform_layer_stu.s_stu_vectors[i].requires_grad = True

        # Clone source model's parameters to target model
        model.fc1.weight.data = self.ncdm_s_net.fc1.weight.clone()
        model.fc1.bias.data = self.ncdm_s_net.fc1.bias.clone()
        model.fc2.weight.data = self.ncdm_s_net.fc2.weight.clone()
        model.fc2.bias.data = self.ncdm_s_net.fc2.bias.clone()
        model.fc3.weight.data = self.ncdm_s_net.fc3.weight.clone()
        model.fc3.bias.data = self.ncdm_s_net.fc3.bias.clone()

        model.prednet_full1.weight.data = self.ncdm_s_net.prednet_full1.weight.data.clone()
        model.prednet_full1.bias.data = self.ncdm_s_net.prednet_full1.bias.data.clone()

        model.prednet_full2.weight.data = self.ncdm_s_net.prednet_full2.weight.data.clone()
        model.prednet_full2.bias.data = self.ncdm_s_net.prednet_full2.bias.data.clone()

        model.prednet_full3.weight.data = self.ncdm_s_net.prednet_full3.weight.data.clone()
        model.prednet_full3.bias.data = self.ncdm_s_net.prednet_full3.bias.data.clone()


    def Target_test(self, model, test_data, device="cpu"):
        t_irt_net = model.to(device)
        t_irt_net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            pred: torch.Tensor = t_irt_net(user_id, item_id)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # Convert probability values to binary labels (0 or 1) to compute F1 score
        y_pred_binary = np.array(y_pred) >= 0.5
        f1 = f1_score(y_true, y_pred_binary)

        # Calculate AUC and accuracy
        auc = roc_auc_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred_binary)

        return auc, accuracy, rmse, f1
