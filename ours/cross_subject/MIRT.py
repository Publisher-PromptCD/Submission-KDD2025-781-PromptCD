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
        self.MLP = SimpleMLP(fc_out_features,10,fc_out_features)
        #self.fc = nn.Linear(fc_out_features, fc_out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        # 将输出展平成一维张量，以便输入全连接层
        x = x.view(x.size(0), -1)  # -1 表示自动推断大小
        x = self.MLP(x)
        # x = self.fc(x)
        return x

class ConvolutionalTransform2(nn.Module):
    def __init__(self, fc_out_features, input_channels=3, output_channels=1, kernel_size=1, stride=1, padding=0):
        super(ConvolutionalTransform2, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size, stride, padding)
        # self.MLP = SimpleMLP(fc_out_features,10,fc_out_features)
        #self.fc = nn.Linear(fc_out_features, fc_out_features)

    def forward(self, x):
        x = self.conv1(x)
        # x = F.relu(x)
        # 将输出展平成一维张量，以便输入全连接层
        x = x.view(x.size(0), -1)  # -1 表示自动推断大小
        # x = self.MLP(x)
        # x = self.fc(x)
        return x

class Transform_Exr(nn.Module):
    def __init__(self, pp_dim, s_ranges):
        super(Transform_Exr, self).__init__()
        self.s_exer_vectors = nn.ParameterList([nn.Parameter(torch.rand(pp_dim)) for _ in range(len(s_ranges))])
        #垂直拼接过一维卷积
        self.Conv1 = ConvolutionalTransform2(pp_dim,input_channels=len(s_ranges))

    def forward(self, x):
        # 将向量加到输入数据上
        #垂直拼接
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
        # 对每个参数进行 Xavier 均匀初始化
        for theta in self.theta:
            nn.init.xavier_uniform_(theta)

        self.prompt_theta = nn.Parameter(torch.randn(self.user_num, self.pp_dim))
        nn.init.xavier_uniform_(self.prompt_theta)

        self.b = nn.Embedding(self.item_num, 1)

        self.a_range = 1

        self.fc1 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)

    def forward(self, user, item):
        # 将参数 prompt_theta 重复 n 次
        prompt_theta_repeated = self.prompt_theta.repeat(len(self.s_ranges), 1)
        # 将列表中的每个元素垂直拼接起来
        theta_concatenated = torch.cat([theta for theta in self.theta], dim=0)
        # 水平拼接两个张量
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
            raise ValueError('ValueError:theta,a,b may contains nan!  The a_range is too large.')
        return self.irf(new_theta, new_a, new_b, **self.irf_kwargs)

    @classmethod
    def irf(cls, theta, a, b, **kwargs):
        return irt2pl(theta, a, b, F=torch)

class Target_MIRTNet(nn.Module):
    def __init__(self, user_num, item_num, latent_dim, pp_dim, s_ranges,a_range, irf_kwargs=None):
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
            raise ValueError('ValueError:theta,a,b may contains nan!  The a_range is too large.')
        return self.irf(new_theta, new_a, new_b, **self.irf_kwargs)

    @classmethod
    def irf(cls, theta, a, b, **kwargs):
        return irt2pl(theta, a, b, F=torch)

class Target_MIRTNet2(nn.Module):
    def __init__(self, user_num, item_num, latent_dim, pp_dim, s_ranges,a_range, irf_kwargs=None):
        super(Target_MIRTNet2, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.pp_dim = pp_dim
        self.s_ranges = s_ranges
        self.latent_dim = latent_dim
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}

        self.a = nn.Embedding(self.item_num, latent_dim)
        self.transform_layer_exr = Transform_Exr(self.pp_dim, self.s_ranges)

        #self.theta = nn.Embedding(self.user_num, latent_dim)
        self.prompt_theta = nn.Embedding(self.user_num, self.pp_dim)
        self.generalize_layer_theta = nn.Linear(self.pp_dim, self.latent_dim)

        self.b = nn.Embedding(self.item_num, 1)

        self.a_range = 1

        self.fc1 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)

    def forward(self, user, item):
        #theta = self.theta(user)
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
            raise ValueError('ValueError:theta,a,b may contains nan!  The a_range is too large.')
        return self.irf(new_theta, new_a, new_b, **self.irf_kwargs)

    @classmethod
    def irf(cls, theta, a, b, **kwargs):
        return irt2pl(theta, a, b, F=torch)

class MIRT():
    def __init__(self, user_num, s_item_num, t_item_num, latent_dim, pp_dim, s_ranges, model_file, target_model_file,a_range=None):
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
                # 保存最佳模型
                if e > 0.001:
                    best_auc = auc
                    consecutive_no_improvement = 0

                    # 保存模型
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

        # 将最佳指标输出到文件
        with open("record.txt", "a") as f:
            f.write(f"Best AUC: {best_auc}, Epoch: {epoch}\n")

    def Target_train(self, model, train_data, test_data=None, epoch=50, device="cpu", lr=0.001, silence=False, patience=5):
        # 迁移训练好的参数
        t_irt_net = model.to(device)
        t_irt_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(t_irt_net.parameters(), lr=lr)

        best_auc = 0.0  # 初始化为较低的值
        best_metrics = None  # 初始化为None
        early_stop_counter = 0  # 早停计数器

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
                    early_stop_counter = 0  # 重置早停计数器
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

        # 计算RMSE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # 将概率值转换为二进制标签（0或1）来计算F1分数
        y_pred_binary = np.array(y_pred) >= 0.5
        f1 = f1_score(y_true, y_pred_binary)

        # 计算AUC和准确率
        auc = roc_auc_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred_binary)

        return auc, accuracy, rmse, f1

    def Transfer_parameters(self, model, s_ranges):
        # 加载模型、迁移参数
        self.s_irt_net.load_state_dict(torch.load(self.model_file))

        model.prompt_theta.weight.data.copy_(
            self.s_irt_net.prompt_theta.data)

        for i in range(len(s_ranges)):
            model.transform_layer_exr.s_exer_vectors[i].data.copy_(
                self.s_irt_net.s_exer_vectors[i].data)
            model.transform_layer_exr.s_exer_vectors[i].requires_grad = True

    def draw_student_distribution2(self):
        # 加载模型
        source_theta = self.s_irt_net.a.cpu()

        # 定义ID范围
        ranges = [range(252, 372), range(0, 137), range(137, 252)]

        # 从每个范围中随机抽取100个ID
        sample_indices = []
        range_labels = []
        for i, r in enumerate(ranges):
            sampled = np.random.choice(list(r), 100, replace=False)
            sample_indices.extend(sampled)
            range_labels.extend([i] * 100)  # 标记这些ID属于哪个范围

        # 取出对应的向量样本
        sample_vectors = source_theta[sample_indices]

        # 将样本转换为numpy数组以便使用sklearn
        sample_vectors_np = sample_vectors.detach().numpy()

        # 使用 t-SNE 降维到2维
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        reduced_vectors = tsne.fit_transform(sample_vectors_np)

        # 自定义标签，可以根据实际需求修改这些标签
        custom_labels = ['Biology (Target)','Mathematics (Source)', 'Physics (Source)']
        # 定义颜色
        colors = ['#4169E1', '#FF7F50', '#2E8B57']

        # 设置字体为 Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'

        # 绘制分布图，根据range_labels进行颜色划分
        plt.figure(figsize=(7, 7))
        ax = plt.gca()
        ax.set_aspect(1,adjustable='datalim')

        for i in range(3):
            idx = np.where(np.array(range_labels) == i)
            plt.scatter(reduced_vectors[idx, 0], reduced_vectors[idx, 1], c=colors[i], label=custom_labels[i])

        plt.title('Exercise-Aspect Cross Domain (Without Prompt)', fontsize=20,fontweight='bold')
        plt.xlabel('Component 1', fontsize=18)
        plt.ylabel('Component 2', fontsize=18)
        plt.legend(fontsize=18,loc='upper right')  # 将图例固定在右上角
        plt.xticks(fontsize=18)  # 设置横坐标数字大小
        plt.yticks(fontsize=18)  # 设置纵坐标数字大小
        plt.show()


    def draw_student_distribution3_1(self):
        temp_vectors = torch.cat(
            [vector.repeat(r[1] - r[0] + 1, 1) for vector, r in zip(self.s_irt_net.s_exer_vectors, self.s_irt_net.s_ranges)], dim=0)
        all_a = torch.cat([temp_vectors, self.s_irt_net.a], dim=1)
        new_a = self.s_irt_net.fc2(all_a)

        # 定义ID范围
        ranges = [range(252, 372), range(0, 137), range(137, 252)]

        # 从每个范围中随机抽取100个ID
        sample_indices = []
        range_labels = []
        for i, r in enumerate(ranges):
            sampled = np.random.choice(list(r), 100, replace=False)
            sample_indices.extend(sampled)
            range_labels.extend([i] * 100)  # 标记这些ID属于哪个范围

        # 取出对应的向量样本
        sample_vectors = new_a[sample_indices]

        # 将样本转换为numpy数组以便使用sklearn
        sample_vectors_np = sample_vectors.detach().numpy()

        # 使用 t-SNE 降维到2维
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        reduced_vectors = tsne.fit_transform(sample_vectors_np)

        # 自定义标签，可以根据实际需求修改这些标签
        custom_labels = ['Biology (Target)','Mathematics (Source)', 'Physics (Source)']
        # 定义颜色
        colors = ['#4169E1', '#FF7F50', '#2E8B57']

        # 设置字体为 Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'

        # 绘制分布图，根据range_labels进行颜色划分
        plt.figure(figsize=(7, 7))
        ax = plt.gca()
        ax.set_aspect(1,adjustable='datalim')

        for i in range(3):
            idx = np.where(np.array(range_labels) == i)
            plt.scatter(reduced_vectors[idx, 0], reduced_vectors[idx, 1], c=colors[i], label=custom_labels[i])

        plt.title('Exercise-Aspect Cross Domain (With Prompt)', fontsize=20,fontweight='bold')
        plt.xlabel('Component 1', fontsize=18)
        plt.ylabel('Component 2', fontsize=18)
        plt.legend(fontsize=18,loc='upper right')  # 将图例固定在右上角
        plt.xticks(fontsize=18)  # 设置横坐标数字大小
        plt.yticks(fontsize=18)  # 设置纵坐标数字大小
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

        # 计算RMSE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # 将概率值转换为二进制标签（0或1）来计算F1分数
        y_pred_binary = np.array(y_pred) >= 0.5
        f1 = f1_score(y_true, y_pred_binary)

        # 计算AUC和准确率
        auc = roc_auc_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred_binary)

        return auc, accuracy, rmse, f1
