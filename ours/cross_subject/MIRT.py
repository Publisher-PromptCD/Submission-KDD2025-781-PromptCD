# coding: utf-8

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class MappingModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MappingModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x

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
        # Flatten the output to a one-dimensional tensor for input into the fully connected layer
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
        # Flatten the output to a one-dimensional tensor for input into the fully connected layer
        x = x.view(x.size(0), -1)  # -1 indicates automatic size inference
        # x = self.MLP(x)
        # x = self.fc(x)
        return x

class Transform_Exr(nn.Module):
    def __init__(self, pp_dim, s_ranges):
        super(Transform_Exr, self).__init__()
        self.s_exer_vectors = nn.ParameterList([nn.Parameter(torch.rand(pp_dim)) for _ in range(len(s_ranges))])
        # Vertically concatenate through 1D convolution
        self.Conv1 = ConvolutionalTransform2(pp_dim, input_channels=len(s_ranges))

    def forward(self, x):
        # Add the vectors to the input data
        # Vertical concatenation
        exr_vector = torch.cat([vector.unsqueeze(0) for vector in self.s_exer_vectors], dim=0)
        new_exr_vector = self.Conv1(exr_vector)
        new_exr_vector = torch.cat([new_exr_vector.expand(x.size(0), -1), x], dim=1)

        return new_exr_vector



def irt2pl(theta, a, b, *, F=np):
    """

    Parameters
    ----------
    theta
    a
    b
    F

    Returns
    -------

    Examples
    --------
    >>> theta = [1, 0.5, 0.3]
    >>> a = [-3, 1, 3]
    >>> b = 0.5
    >>> irt2pl(theta, a, b) # doctest: +ELLIPSIS
    0.109...
    >>> theta = [[1, 0.5, 0.3], [2, 1, 0]]
    >>> a = [[-3, 1, 3], [-3, 1, 3]]
    >>> b = [0.5, 0.5]
    >>> irt2pl(theta, a, b) # doctest: +ELLIPSIS
    array([0.109..., 0.004...])
    """
    return 1 / (1 + F.exp(- F.sum(F.multiply(a, theta), axis=-1) + b))


class Source_MIRTNet(nn.Module):
    def __init__(self, user_num, item_num, latent_dim, pp_dim, s_ranges, a_range, irf_kwargs=None):
        super(Source_MIRTNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.pp_dim = pp_dim
        self.s_ranges = s_ranges
        self.latent_dim = latent_dim
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}

        self.a = nn.Parameter(torch.rand((self.item_num, self.latent_dim)))
        nn.init.xavier_uniform_(self.a)
        self.s_exer_vectors = nn.ParameterList([nn.Parameter(torch.rand(self.pp_dim)) for _ in range(len(s_ranges))])

        self.theta = nn.ParameterList([nn.Parameter(torch.randn(self.user_num, self.latent_dim))
                                       for _ in range(len(s_ranges))])
        # Perform Xavier uniform initialization for each parameter
        for theta in self.theta:
            nn.init.xavier_uniform_(theta)

        self.prompt_theta = nn.Parameter(torch.randn(self.user_num, self.pp_dim))
        nn.init.xavier_uniform_(self.prompt_theta)

        self.b = nn.Embedding(self.item_num, 1)

        self.a_range = 1

        self.fc1 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)

    def forward(self, user, item):
        # Repeat the prompt_theta parameter n times
        prompt_theta_repeated = self.prompt_theta.repeat(len(self.s_ranges), 1)
        # Vertically concatenate two tensors
        theta_concatenated = torch.cat([theta for theta in self.theta], dim=0)
        # Horizontally concatenate two tensors
        new_theta = torch.cat([prompt_theta_repeated, theta_concatenated], dim=1)
        new_theta = torch.sigmoid(torch.index_select(new_theta, dim=0, index=user))
        new_theta = self.fc1(new_theta)

        temp_vectors = torch.cat(
            [vector.repeat(r[1] - r[0] + 1, 1) for vector, r in zip(self.s_exer_vectors, self.s_ranges)], dim=0)
        all_a = torch.cat([temp_vectors, self.a], dim=1)
        new_a = torch.sigmoid(torch.index_select(all_a, dim=0, index=item))
        new_a = self.fc2(new_a)

        new_b = self.b(item)
        new_b = torch.squeeze(new_b, dim=-1)
        if self.a_range is not None:
            new_a = self.a_range * torch.sigmoid(new_a)
            new_b = self.a_range * torch.sigmoid(new_b)
            new_theta = self.a_range * torch.sigmoid(new_theta)
        else:
            new_a = F.softplus(new_a)
    #------------------------------

        if torch.max(new_theta != new_theta) or torch.max(new_a != new_a) or torch.max(new_b != new_b):  # pragma: no cover
            raise ValueError('ValueError: theta, a, b may contain NaN! The a_range is too large.')
        return self.irf(new_theta, new_a, new_b, **self.irf_kwargs)

    @classmethod
    def irf(cls, theta, a, b, **kwargs):
        return irt2pl(theta, a, b, F=torch)

class Target_MIRTNet(nn.Module):
    def __init__(self, user_num, item_num, latent_dim, pp_dim, s_ranges, a_range, irf_kwargs=None):
        super(Target_MIRTNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.pp_dim = pp_dim
        self.s_ranges = s_ranges
        self.latent_dim = latent_dim
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}

        self.a = nn.Embedding(self.item_num, latent_dim)
        self.transform_layer_exr = Transform_Exr(self.pp_dim, self.s_ranges)

        self.theta = nn.Embedding(self.user_num, latent_dim)
        self.prompt_theta = nn.Embedding(self.user_num, self.pp_dim)

        self.b = nn.Embedding(self.item_num, 1)

        self.a_range = 1

        self.fc1 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)

    def forward(self, user, item):
        theta = self.theta(user)
        p_theta = self.prompt_theta(user)
        new_theta = torch.cat([p_theta, theta], dim=1)
        new_theta = self.fc1(new_theta)

        a = self.a(item)
        new_a = self.transform_layer_exr(a)
        new_a = self.fc2(new_a)

        new_b = self.b(item)
        new_b = torch.squeeze(new_b, dim=-1)
        if self.a_range is not None:
            new_a = self.a_range * torch.sigmoid(new_a)
            new_b = self.a_range * torch.sigmoid(new_b)
            new_theta = self.a_range * torch.sigmoid(new_theta)
        else:
            new_a = F.softplus(new_a)

        if torch.max(new_theta != new_theta) or torch.max(new_a != new_a) or torch.max(new_b != new_b):  # pragma: no cover
            raise ValueError('ValueError: theta, a, b may contain NaN! The a_range is too large.')
        return self.irf(new_theta, new_a, new_b, **self.irf_kwargs)

    @classmethod
    def irf(cls, theta, a, b, **kwargs):
        return irt2pl(theta, a, b, F=torch)

class Target_MIRTNet2(nn.Module):
    def __init__(self, user_num, item_num, latent_dim, pp_dim, s_ranges, a_range, irf_kwargs=None):
        super(Target_MIRTNet2, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.pp_dim = pp_dim
        self.s_ranges = s_ranges
        self.latent_dim = latent_dim
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}

        self.a = nn.Embedding(self.item_num, latent_dim)
        self.transform_layer_exr = Transform_Exr(self.pp_dim, self.s_ranges)

        # self.theta = nn.Embedding(self.user_num, latent_dim)
        self.prompt_theta = nn.Embedding(self.user_num, self.pp_dim)
        self.generalize_layer_theta = nn.Linear(self.pp_dim, self.latent_dim)

        self.b = nn.Embedding(self.item_num, 1)

        self.a_range = 1

        self.fc1 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)

    def forward(self, user, item):
        # theta = self.theta(user)
        p_theta = self.prompt_theta(user)
        theta = self.generalize_layer_theta(p_theta)
        new_theta = torch.cat([p_theta, theta], dim=1)
        new_theta = self.fc1(new_theta)

        a = self.a(item)
        new_a = self.transform_layer_exr(a)
        new_a = self.fc2(new_a)

        new_b = self.b(item)
        new_b = torch.squeeze(new_b, dim=-1)
        if self.a_range is not None:
            new_a = self.a_range * torch.sigmoid(new_a)
            new_b = self.a_range * torch.sigmoid(new_b)
            new_theta = self.a_range * torch.sigmoid(new_theta)
        else:
            new_a = F.softplus(new_a)

        if torch.max(new_theta != new_theta) or torch.max(new_a != new_a) or torch.max(new_b != new_b):  # pragma: no cover
            raise ValueError('ValueError: theta, a, b may contain NaN! The a_range is too large.')
        return self.irf(new_theta, new_a, new_b, **self.irf_kwargs)

    @classmethod
    def irf(cls, theta, a, b, **kwargs):
        return irt2pl(theta, a, b, F=torch)

class MIRT():
    def __init__(self, user_num, s_item_num, t_item_num, latent_dim, pp_dim, s_ranges, model_file, target_model_file, a_range=None):
        super(MIRT, self).__init__()
        self.model_file = model_file
        self.pp_dim = pp_dim
        self.latent_dim = latent_dim
        self.target_model_file = target_model_file
        self.s_irt_net = Source_MIRTNet(user_num, s_item_num, latent_dim, pp_dim, s_ranges, a_range)
        self.t_irt_net = Target_MIRTNet(user_num, t_item_num, latent_dim, pp_dim, s_ranges, a_range)
        self.t_irt_net2 = Target_MIRTNet2(user_num, t_item_num, latent_dim, pp_dim, s_ranges, a_range)

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
                user_id, item_id, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                y: torch.Tensor = y.to(device)

                pred: torch.Tensor = self.s_irt_net(user_id, item_id)
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            average_loss = float(np.mean(epoch_losses))
            print("[Epoch %d] average loss: %.6f" % (epoch, average_loss))

            if test_data is not None:
                auc, accuracy = self.Source_net_eval(test_data, device=device)
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

        # Output the best metric to a file
        with open("record.txt", "a") as f:
            f.write(f"Best AUC: {best_auc}, Epoch: {epoch}\n")

    def Target_train(self, model, train_data, test_data=None, epoch=50, device="cpu", lr=0.001, silence=False, patience=5):
        # Transfer trained parameters
        t_irt_net = model.to(device)
        t_irt_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(t_irt_net.parameters(), lr=lr)

        best_auc = 0.0  # Initialize to a low value
        best_metrics = None  # Initialize to None
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
                auc, accuracy, rmse, f1 = self.Target_net_eval(model, test_data, device=device)
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
        #[92, 17, 49, 103, 56, 24, 31, 108, 89, 12]

    def Source_net_eval(self, test_data, device="cpu"):
        self.s_irt_net = self.s_irt_net.to(device)
        self.s_irt_net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            pred: torch.Tensor = self.s_irt_net(user_id, item_id)
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

        # Convert probability values to binary labels (0 or 1) to compute F1 score
        y_pred_binary = np.array(y_pred) >= 0.5
        f1 = f1_score(y_true, y_pred_binary)

        # Calculate AUC and accuracy
        auc = roc_auc_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred_binary)

        return auc, accuracy, rmse, f1

    def Transfer_parameters(self, model, s_ranges):
        # Load model and transfer parameters
        self.s_irt_net.load_state_dict(torch.load(self.model_file))

        model.prompt_theta.weight.data.copy_(
            self.s_irt_net.prompt_theta.data)

        for i in range(len(s_ranges)):
            model.transform_layer_exr.s_exer_vectors[i].data.copy_(
                self.s_irt_net.s_exer_vectors[i].data)
            model.transform_layer_exr.s_exer_vectors[i].requires_grad = True

        # Clone source model's parameters to target model
        model.fc1.weight.data = self.s_irt_net.fc1.weight.clone()
        model.fc1.bias.data = self.s_irt_net.fc1.bias.clone()
        model.fc2.weight.data = self.s_irt_net.fc2.weight.clone()
        model.fc2.bias.data = self.s_irt_net.fc2.bias.clone()
        model.fc3.weight.data = self.s_irt_net.fc3.weight.clone()
        model.fc3.bias.data = self.s_irt_net.fc3.bias.clone()

        model.prednet_full1.weight.data = self.s_irt_net.prednet_full1.weight.data.clone()
        model.prednet_full1.bias.data = self.s_irt_net.prednet_full1.bias.data.clone()

        model.prednet_full2.weight.data = self.s_irt_net.prednet_full2.weight.data.clone()
        model.prednet_full2.bias.data = self.s_irt_net.prednet_full2.bias.data.clone()

        model.prednet_full3.weight.data = self.s_irt_net.prednet_full3.weight.data.clone()
        model.prednet_full3.bias.data = self.s_irt_net.prednet_full3.bias.data.clone()


    def draw_student_distribution2(self):
        # Load the model
        source_theta = self.s_irt_net.a.cpu()

        # Define ID ranges
        ranges = [range(252, 372), range(0, 137), range(137, 252)]

        # Randomly sample 100 IDs from each range
        sample_indices = []
        range_labels = []
        for i, r in enumerate(ranges):
            sampled = np.random.choice(list(r), 100, replace=False)
            sample_indices.extend(sampled)
            range_labels.extend([i] * 100)  # Label these IDs as belonging to which range

        # Extract the corresponding vector samples
        sample_vectors = source_theta[sample_indices]

        # Convert samples to numpy array for use with sklearn
        sample_vectors_np = sample_vectors.detach().numpy()

        # Use t-SNE to reduce dimensions to 2
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        reduced_vectors = tsne.fit_transform(sample_vectors_np)

        # Custom labels, can be modified based on actual needs
        custom_labels = ['Biology (Target)', 'Mathematics (Source)', 'Physics (Source)']
        # Define colors
        colors = ['#4169E1', '#FF7F50', '#2E8B57']

        # Set font to Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'

        # Plot the distribution, color-coded based on range_labels
        plt.figure(figsize=(7, 7))
        ax = plt.gca()
        ax.set_aspect(1, adjustable='datalim')

        for i in range(3):
            idx = np.where(np.array(range_labels) == i)
            plt.scatter(reduced_vectors[idx, 0], reduced_vectors[idx, 1], c=colors[i], label=custom_labels[i])

        plt.title('Exercise-Aspect Cross Domain (Without Prompt)', fontsize=20, fontweight='bold')
        plt.xlabel('Component 1', fontsize=18)
        plt.ylabel('Component 2', fontsize=18)
        plt.legend(fontsize=18, loc='upper right')  # Fix the legend in the upper right corner
        plt.xticks(fontsize=18)  # Set the size of the x-axis numbers
        plt.yticks(fontsize=18)  # Set the size of the y-axis numbers
        plt.show()


    def draw_student_distribution3_1(self):
        temp_vectors = torch.cat(
            [vector.repeat(r[1] - r[0] + 1, 1) for vector, r in zip(self.s_irt_net.s_exer_vectors, self.s_irt_net.s_ranges)], dim=0)
        all_a = torch.cat([temp_vectors, self.s_irt_net.a], dim=1)
        new_a = self.s_irt_net.fc2(all_a)

        # Define ID ranges
        ranges = [range(252, 372), range(0, 137), range(137, 252)]

        # Randomly sample 100 IDs from each range
        sample_indices = []
        range_labels = []
        for i, r in enumerate(ranges):
            sampled = np.random.choice(list(r), 100, replace=False)
            sample_indices.extend(sampled)
            range_labels.extend([i] * 100)  # Label these IDs as belonging to which range

        # Extract the corresponding vector samples
        sample_vectors = new_a[sample_indices]

        # Convert samples to numpy array for use with sklearn
        sample_vectors_np = sample_vectors.detach().numpy()

        # Use t-SNE to reduce dimensions to 2
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        reduced_vectors = tsne.fit_transform(sample_vectors_np)

        # Custom labels, can be modified based on actual needs
        custom_labels = ['Biology (Target)', 'Mathematics (Source)', 'Physics (Source)']
        # Define colors
        colors = ['#4169E1', '#FF7F50', '#2E8B57']

        # Set font to Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'

        # Plot the distribution, color-coded based on range_labels
        plt.figure(figsize=(7, 7))
        ax = plt.gca()
        ax.set_aspect(1, adjustable='datalim')

        for i in range(3):
            idx = np.where(np.array(range_labels) == i)
            plt.scatter(reduced_vectors[idx, 0], reduced_vectors[idx, 1], c=colors[i], label=custom_labels[i])

        plt.title('Exercise-Aspect Cross Domain (With Prompt)', fontsize=20, fontweight='bold')
        plt.xlabel('Component 1', fontsize=18)
        plt.ylabel('Component 2', fontsize=18)
        plt.legend(fontsize=18, loc='upper right')  # Fix the legend in the upper right corner
        plt.xticks(fontsize=18)  # Set the size of the x-axis numbers
        plt.yticks(fontsize=18)  # Set the size of the y-axis numbers
        plt.show()


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
