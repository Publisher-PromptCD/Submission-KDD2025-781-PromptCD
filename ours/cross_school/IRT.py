# coding: utf-8

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvolutionalTransform(nn.Module):
    def __init__(self, fc_out_features, input_channels=3, output_channels=1, kernel_size=1, stride=1, padding=0):
        super(ConvolutionalTransform, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size, stride, padding)
        self.MLP = SimpleMLP(fc_out_features, 10, fc_out_features)
        # self.fc = nn.Linear(fc_out_features, fc_out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        # Flatten the output to a one-dimensional tensor for the fully connected layer
        x = x.view(x.size(0), -1)  # -1 indicates automatic size inference
        x = self.MLP(x)
        # x = self.fc(x)
        return x


class ConvolutionalTransform2(nn.Module):
    def __init__(self, fc_out_features, input_channels=3, output_channels=1, kernel_size=1, stride=1, padding=0):
        super(ConvolutionalTransform2, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size, stride, padding)
        # self.MLP = SimpleMLP(fc_out_features,10,fc_out_features)
        # self.fc = nn.Linear(fc_out_features, fc_out_features)

    def forward(self, x):
        x = self.conv1(x)
        # x = F.relu(x)
        # Flatten the output to a one-dimensional tensor for the fully connected layer
        x = x.view(x.size(0), -1)  # -1 indicates automatic size inference
        # x = self.MLP(x)
        # x = self.fc(x)
        return x


class Transform_stu(nn.Module):
    def __init__(self, pp_dim, s_ranges):
        super(Transform_stu, self).__init__()
        self.s_stu_vectors = nn.ParameterList([nn.Parameter(torch.rand(pp_dim)) for _ in range(len(s_ranges))])
        # Apply one-dimensional convolution after vertical concatenation
        self.Conv1 = ConvolutionalTransform2(pp_dim, input_channels=len(s_ranges))

    def forward(self, x):
        # Concatenate vectors vertically
        stu_vector = torch.cat([vector.unsqueeze(0) for vector in self.s_stu_vectors], dim=0)
        new_stu_vector = self.Conv1(stu_vector)
        new_stu_vector = torch.cat([new_stu_vector.expand(x.size(0), -1), x], dim=1)

        return new_stu_vector


class Source_MIRTNet(nn.Module):
    def __init__(self, user_num, item_num, latent_dim, pp_dim, s_ranges, a_range, irf_kwargs=None):
        #------------------------------------------------------- Convert theta to b, and b to theta
        super(Source_MIRTNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.pp_dim = pp_dim
        self.s_ranges = s_ranges
        self.latent_dim = latent_dim
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}

        self.theta = nn.Parameter(torch.rand((self.user_num, self.latent_dim)))
        nn.init.xavier_uniform_(self.theta)
        self.s_stu_vectors = nn.ParameterList([nn.Parameter(torch.rand(self.pp_dim)) for _ in range(len(s_ranges))])

        self.b = nn.ParameterList([nn.Parameter(torch.randn(self.item_num, self.latent_dim))
                                   for _ in range(len(s_ranges))])
        # Initialize each parameter with Xavier uniform initialization
        for b in self.b:
            nn.init.xavier_uniform_(b)
        self.prompt_b = nn.Parameter(torch.randn(self.item_num, self.pp_dim))
        nn.init.xavier_uniform_(self.prompt_b)

        self.a = nn.ParameterList([nn.Parameter(torch.randn(self.item_num, self.latent_dim))
                                   for _ in range(len(s_ranges))])
        # Initialize each parameter with Xavier uniform initialization
        for a in self.a:
            nn.init.xavier_uniform_(a)
        self.prompt_a = nn.Parameter(torch.randn(self.item_num, self.pp_dim))
        nn.init.xavier_uniform_(self.prompt_a)

        self.c = nn.ParameterList([nn.Parameter(torch.randn(self.item_num, self.latent_dim))
                                   for _ in range(len(s_ranges))])
        # Initialize each parameter with Xavier uniform initialization
        for c in self.c:
            nn.init.xavier_uniform_(c)
        self.prompt_c = nn.Parameter(torch.randn(self.item_num, self.pp_dim))
        nn.init.xavier_uniform_(self.prompt_c)

        self.a_range = 1  # Set hyperparameter here
        self.value_range = 8

        self.fc1 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)
        self.fc3 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)
        self.fc4 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)

    def forward(self, user, item, item2):  #------------------- Convert theta to b, and b to theta
        # Repeat the prompt_b parameter n times
        prompt_b_repeated = self.prompt_b.repeat(len(self.s_ranges), 1)
        # Concatenate each element in the list vertically
        b_concatenated = torch.cat([b for b in self.b], dim=0)
        # Concatenate two tensors horizontally
        new_b = torch.cat([prompt_b_repeated, b_concatenated], dim=1)
        new_b = torch.index_select(new_b, dim=0, index=item2)
        new_b = self.fc1(new_b)
        new_b = torch.squeeze(new_b, dim=-1)

        # Repeat the prompt_a parameter n times
        prompt_a_repeated = self.prompt_a.repeat(len(self.s_ranges), 1)
        # Concatenate each element in the list vertically
        a_concatenated = torch.cat([a for a in self.a], dim=0)
        # Concatenate two tensors horizontally
        new_a = torch.cat([prompt_a_repeated, a_concatenated], dim=1)
        new_a = torch.index_select(new_a, dim=0, index=item2)
        new_a = self.fc2(new_a)
        new_a = torch.squeeze(new_a, dim=-1)

        # Repeat the prompt_c parameter n times
        prompt_c_repeated = self.prompt_c.repeat(len(self.s_ranges), 1)
        # Concatenate each element in the list vertically
        c_concatenated = torch.cat([c for c in self.c], dim=0)
        # Concatenate two tensors horizontally
        new_c = torch.cat([prompt_c_repeated, c_concatenated], dim=1)
        new_c = torch.index_select(new_c, dim=0, index=item2)
        new_c = self.fc3(new_c)
        new_c = torch.squeeze(new_c, dim=-1)
        new_c = torch.sigmoid(new_c)

        temp_vectors = torch.cat(
            [vector.repeat(r[1] - r[0] + 1, 1) for vector, r in zip(self.s_stu_vectors, self.s_ranges)], dim=0)
        all_theta = torch.cat([temp_vectors, self.theta], dim=1)
        new_theta = torch.index_select(all_theta, dim=0, index=user)
        new_theta = self.fc4(new_theta)
        new_theta = torch.squeeze(new_theta, dim=-1)

        if self.value_range is not None:
            new_theta = self.value_range * (torch.sigmoid(new_theta) - 0.5)
            new_b = self.value_range * (torch.sigmoid(new_b) - 0.5)
        if self.a_range is not None:
            new_a = self.a_range * torch.sigmoid(new_a)
        else:
            new_a = F.softplus(new_a)

        # ------------------------------
        if torch.max(new_theta != new_theta) or torch.max(new_a != new_a) or torch.max(new_b != new_b):  # pragma: no cover
            raise ValueError('ValueError: theta, a, b may contain nan! The a_range is too large.')
        return self.irf(new_theta, new_a, new_b, new_c, **self.irf_kwargs)

    @classmethod
    def irf(cls, theta, a, b, c, **kwargs):
        D = 1.702
        F = torch
        return c + (1 - c) / (1 + F.exp(-D * a * (theta - b)))


class Target_MIRTNet(nn.Module):
    def __init__(self, user_num, item_num, latent_dim, pp_dim, s_ranges, a_range=None, irf_kwargs=None):
        super(Target_MIRTNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.pp_dim = pp_dim
        self.s_ranges = s_ranges
        self.latent_dim = latent_dim
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}

        self.theta = nn.Embedding(self.user_num, self.latent_dim)
        self.transform_layer_stu = Transform_stu(self.pp_dim, self.s_ranges)

        self.b = nn.Embedding(self.item_num, self.latent_dim)
        self.prompt_b = nn.Embedding(self.item_num, self.pp_dim)

        self.a = nn.Embedding(self.item_num, self.latent_dim)
        self.prompt_a = nn.Embedding(self.item_num, self.pp_dim)

        self.c = nn.Embedding(self.item_num, self.latent_dim)
        self.prompt_c = nn.Embedding(self.item_num, self.pp_dim)

        self.a_range = 1
        self.value_range = 8

        self.fc1 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)
        self.fc3 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)
        self.fc4 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)

    def forward(self, user, item):
        b = self.b(item)
        p_b = self.prompt_b(item)
        new_b = torch.cat([p_b, b], dim=1)
        new_b = self.fc1(new_b)
        new_b = torch.squeeze(new_b, dim=-1)

        a = self.a(item)
        p_a = self.prompt_a(item)
        new_a = torch.cat([p_a, a], dim=1)
        new_a = self.fc2(new_a)
        new_a = torch.squeeze(new_a, dim=-1)

        c = self.c(item)
        p_c = self.prompt_c(item)
        new_c = torch.cat([p_c, c], dim=1)
        new_c = self.fc3(new_c)
        new_c = torch.squeeze(new_c, dim=-1)
        new_c = torch.sigmoid(new_c)

        theta = self.theta(user)
        new_theta = self.transform_layer_stu(theta)
        new_theta = self.fc4(new_theta)
        new_theta = torch.squeeze(new_theta, dim=-1)

        if self.value_range is not None:
            new_theta = self.value_range * (torch.sigmoid(new_theta) - 0.5)
            new_b = self.value_range * (torch.sigmoid(new_b) - 0.5)
        if self.a_range is not None:
            new_a = self.a_range * torch.sigmoid(new_a)
        else:
            new_a = F.softplus(new_a)

        if torch.max(new_theta != new_theta) or torch.max(new_a != new_a) or torch.max(new_b != new_b):  # pragma: no cover
            raise ValueError('ValueError: theta, a, b may contain nan! The a_range is too large.')

        return self.irf(new_theta, new_a, new_b, new_c, **self.irf_kwargs)

    @classmethod
    def irf(cls, theta, a, b, c, **kwargs):
        D = 1.702
        F = torch
        return c + (1 - c) / (1 + F.exp(-D * a * (theta - b)))


class Target_MIRTNet2(nn.Module):  #-------------------
    def __init__(self, user_num, item_num, latent_dim, pp_dim, s_ranges, a_range=None, irf_kwargs=None):
        super(Target_MIRTNet2, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.pp_dim = pp_dim
        self.s_ranges = s_ranges
        self.latent_dim = latent_dim
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}

        self.theta = nn.Embedding(self.user_num, self.latent_dim)
        self.transform_layer_stu = Transform_stu(self.pp_dim, self.s_ranges)

        # self.b = nn.Embedding(self.item_num, self.latent_dim)
        self.generalize_layer_b = nn.Linear(self.pp_dim, self.latent_dim)
        self.prompt_b = nn.Embedding(self.item_num, self.pp_dim)

        # self.a = nn.Embedding(self.item_num, self.latent_dim)
        self.generalize_layer_a = nn.Linear(self.pp_dim, self.latent_dim)
        self.prompt_a = nn.Embedding(self.item_num, self.pp_dim)

        # self.c = nn.Embedding(self.item_num, self.latent_dim)
        self.generalize_layer_c = nn.Linear(self.pp_dim, self.latent_dim)
        self.prompt_c = nn.Embedding(self.item_num, self.pp_dim)

        self.a_range = 1
        self.value_range = 8

        self.fc1 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)
        self.fc3 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)
        self.fc4 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)

    def forward(self, user, item):
        # b = self.b(item)
        p_b = self.prompt_b(item)
        b = self.generalize_layer_b(p_b)
        new_b = torch.cat([p_b, b], dim=1)
        new_b = self.fc1(new_b)
        new_b = torch.squeeze(new_b, dim=-1)

        # a = self.a(item)
        p_a = self.prompt_a(item)
        a = self.generalize_layer_a(p_a)
        new_a = torch.cat([p_a, a], dim=1)
        new_a = self.fc2(new_a)
        new_a = torch.squeeze(new_a, dim=-1)

        # c = self.c(item)
        p_c = self.prompt_c(item)
        c = self.generalize_layer_c(p_c)
        new_c = torch.cat([p_c, c], dim=1)
        new_c = self.fc3(new_c)
        new_c = torch.squeeze(new_c, dim=-1)
        new_c = torch.sigmoid(new_c)

        theta = self.theta(user)
        new_theta = self.transform_layer_stu(theta)
        new_theta = self.fc4(new_theta)
        new_theta = torch.squeeze(new_theta, dim=-1)

        if self.value_range is not None:
            new_theta = self.value_range * (torch.sigmoid(new_theta) - 0.5)
            new_b = self.value_range * (torch.sigmoid(new_b) - 0.5)
        if self.a_range is not None:
            new_a = self.a_range * torch.sigmoid(new_a)
        else:
            new_a = F.softplus(new_a)

        if torch.max(new_theta != new_theta) or torch.max(new_a != new_a) or torch.max(new_b != new_b):  # pragma: no cover
            raise ValueError('ValueError: theta, a, b may contain nan! The a_range is too large.')

        return self.irf(new_theta, new_a, new_b, new_c, **self.irf_kwargs)


class IRT():
    def __init__(self, s_user_num, t_user_num, item_num, latent_dim, pp_dim, s_ranges, model_file, target_model_file, a_range=None):
        super(IRT, self).__init__()
        self.model_file = model_file
        self.target_model_file = target_model_file
        self.s_irt_net = Source_MIRTNet(s_user_num, item_num, latent_dim, pp_dim, s_ranges, a_range)
        self.t_irt_net = Target_MIRTNet(t_user_num, item_num, latent_dim, pp_dim, s_ranges, a_range)
        self.t_irt_net2 = Target_MIRTNet2(t_user_num, item_num, latent_dim, pp_dim, s_ranges, a_range)

    def Source_train(self, train_data, test_data=None, max_epochs=50, early_stopping_patience=5, device="cpu",
                     lr=0.001, silence=False):
        self.s_irt_net = self.s_irt_net.to(device)
        self.s_irt_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.s_irt_net.parameters(), lr=lr)

        epoch = 0
        best_auc = 0.0
        consecutive_no_improvement = 0

        while max_epochs is None or epoch < max_epochs:
            epoch_losses = []
            batch_count = 0

            for batch_data in tqdm(train_data, "Epoch %s" % epoch):
                batch_count += 1
                user_id, item_id, item_id2, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                item_id2: torch.Tensor = item_id2.to(device)
                y: torch.Tensor = y.to(device)

                pred: torch.Tensor = self.s_irt_net(user_id, item_id, item_id2)
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            average_loss = float(np.mean(epoch_losses))
            print("[Epoch %d] average loss: %.6f" % (epoch, average_loss))

            if test_data is not None:
                auc, accuracy = self.Source_net_eval(self.s_irt_net, test_data, device=device)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (epoch, auc, accuracy))
                e = auc - best_auc
                # Save the best model
                if e > 0.001:
                    best_auc = auc
                    consecutive_no_improvement = 0

                    # Save the model
                    torch.save(self.s_irt_net.state_dict(), self.model_file)
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

        # Output the best metrics to the file
        with open("record.txt", "a") as f:
            f.write(f"Best AUC: {best_auc}, Epoch: {epoch}\n")

    def Target_train(self, model, train_data, test_data=None, epoch=50, device="cpu", lr=0.001, silence=False, patience=5):
        # Transfer the trained parameters
        t_irt_net = model.to(device)
        t_irt_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(t_irt_net.parameters(), lr=lr)

        best_auc = 0.0  # Initialize with a low value
        best_metrics = None  # Initialize as None
        early_stop_counter = 0  # Early stopping counter

        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                user_id, item_id, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                y: torch.Tensor = y.to(device)
                pred: torch.Tensor = t_irt_net(user_id, item_id)
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            average_loss = np.mean(epoch_losses)
            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(average_loss)))

            if test_data is not None:
                auc, accuracy, rmse, f1 = self.Target_net_eval(t_irt_net, test_data, device=device)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f, RMSE: %.6f, F1: %.6f" % (epoch_i, auc, accuracy, rmse, f1))

                e = auc - best_auc
                # Update best metrics if current metrics are better
                if e > 0.0001:
                    best_auc = auc
                    best_metrics = (auc, accuracy, rmse, f1)
                    early_stop_counter = 0  # Reset early stopping counter
                    torch.save(t_irt_net.state_dict(), self.target_model_file)
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

    def Source_net_eval(self, model, test_data, device="cpu"):
        s_irt_net = model.to(device)
        s_irt_net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, item_id2, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            item_id2: torch.Tensor = item_id2.to(device)
            pred: torch.Tensor = s_irt_net(user_id, item_id, item_id2)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def Target_net_eval(self, model, test_data, device="cpu"):
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

        # Convert probability values to binary labels (0 or 1) to calculate F1 score
        y_pred_binary = np.array(y_pred) >= 0.5
        f1 = f1_score(y_true, y_pred_binary)

        # Calculate AUC and accuracy
        auc = roc_auc_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred_binary)

        return auc, accuracy, rmse, f1

    def Transfer_parameters(self, model, s_ranges):
        # Load the model and transfer parameters
        self.s_irt_net.load_state_dict(torch.load(self.model_file))

        model.prompt_a.weight.data.copy_(self.s_irt_net.prompt_a.data)

        model.prompt_b.weight.data.copy_(self.s_irt_net.prompt_b.data)

        model.prompt_c.weight.data.copy_(self.s_irt_net.prompt_c.data)

        for i in range(len(s_ranges)):
            model.transform_layer_stu.s_stu_vectors[i].data.copy_(
                self.s_irt_net.s_stu_vectors[i].data)
            model.transform_layer_stu.s_stu_vectors[i].requires_grad = True

    def draw_student_distribution(self):
        # Load the model and transfer parameters
        self.s_irt_net.load_state_dict(torch.load(self.model_file))

        self.t_irt_net2.prompt_a.weight.data.copy_(self.s_irt_net.prompt_a.data)

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

        # Convert probability values to binary labels (0 or 1) to calculate F1 score
        y_pred_binary = np.array(y_pred) >= 0.5
        f1 = f1_score(y_true, y_pred_binary)

        # Calculate AUC and accuracy
        auc = roc_auc_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred_binary)

        return auc, accuracy, rmse, f1

