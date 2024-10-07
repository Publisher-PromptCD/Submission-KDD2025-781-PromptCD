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
        # self.fc = nn.Linear(fc_out_features, fc_out_features)

    def forward(self, x):
        x = self.conv1(x)
        # x = F.relu(x)
        # 将输出展平成一维张量，以便输入全连接层
        x = x.view(x.size(0), -1)  # -1 表示自动推断大小
        # x = self.MLP(x)
        # x = self.fc(x)
        return x

class ConvolutionalTransform3(nn.Module):
    def __init__(self, fc_out_features, input_channels=3, output_channels=1, kernel_size=1, stride=1, padding=0):
        super(ConvolutionalTransform3, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size, stride, padding)
        # self.MLP = SimpleMLP(fc_out_features,10,fc_out_features)
        self.fc = nn.Linear(fc_out_features, fc_out_features)

    def forward(self, x):
        x = self.conv1(x)
        #x = F.relu(x)
        # 将输出展平成一维张量，以便输入全连接层
        x = x.view(x.size(0), -1)  # -1 表示自动推断大小
        # x = self.MLP(x)
        x = self.fc(x)
        return x

class Transform_stu(nn.Module):
    def __init__(self, pp_dim, s_ranges):
        super(Transform_stu, self).__init__()
        self.s_stu_vectors = nn.ParameterList([nn.Parameter(torch.rand(pp_dim)) for _ in range(len(s_ranges))])
        #垂直拼接过一维卷积
        self.Conv1 = ConvolutionalTransform2(pp_dim,input_channels=len(s_ranges))

    def forward(self, x):
        # 将向量加到输入数据上
        #垂直拼接
        stu_vector = torch.cat([vector.unsqueeze(0) for vector in self.s_stu_vectors], dim=0)
        new_stu_vector = self.Conv1(stu_vector)
        new_stu_vector = torch.cat([new_stu_vector.expand(x.size(0), -1), x], dim=1)

        return new_stu_vector

def irt2pl(theta, a, b, *, F=np):
    return 1 / (1 + F.exp(- F.sum(F.multiply(a, theta), axis=-1) + b))

class Source_MIRTNet(nn.Module): #------------------a(item), theta(user)互换
    def __init__(self, user_num, item_num, latent_dim, pp_dim, s_ranges, a_range, irf_kwargs=None):
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

        self.a = nn.ParameterList([nn.Parameter(torch.randn(self.item_num, self.latent_dim))
                                       for _ in range(len(s_ranges))])
        # 对每个参数进行 Xavier 均匀初始化
        for a in self.a:
            nn.init.xavier_uniform_(a)

        self.prompt_a = nn.Parameter(torch.randn(self.item_num, self.pp_dim))
        nn.init.xavier_uniform_(self.prompt_a)

        self.b = nn.ParameterList([nn.Parameter(torch.randn(self.item_num, 1))
                                       for _ in range(len(s_ranges))])
        # 对每个参数进行 Xavier 均匀初始化
        for b in self.b:
            nn.init.xavier_uniform_(b)

        self.prompt_b = nn.Parameter(torch.randn(self.item_num, self.pp_dim))
        nn.init.xavier_uniform_(self.prompt_b)

        self.a_range = 1

        self.fc1 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)
        self.fc3 = nn.Linear(self.pp_dim + 1, 1)

    def forward(self, user, item, item2):
        # 将参数 prompt_a 重复 n 次
        prompt_a_repeated = self.prompt_a.repeat(len(self.s_ranges), 1)
        # 将列表中的每个元素垂直拼接起来
        a_concatenated = torch.cat([a for a in self.a], dim=0)
        # 水平拼接两个张量
        new_a = torch.cat([prompt_a_repeated, a_concatenated], dim=1)
        new_a = torch.index_select(new_a, dim=0, index=item2)
        new_a = self.fc1(new_a)

        temp_vectors = torch.cat(
            [vector.repeat(r[1] - r[0] + 1, 1) for vector, r in zip(self.s_stu_vectors, self.s_ranges)], dim=0)
        all_theta = torch.cat([temp_vectors, self.theta], dim=1)
        new_theta = torch.index_select(all_theta, dim=0, index=user)
        new_theta = self.fc2(new_theta)

        # 将参数 prompt_a 重复 n 次
        prompt_b_repeated = self.prompt_b.repeat(len(self.s_ranges), 1)
        # 将列表中的每个元素垂直拼接起来
        b_concatenated = torch.cat([b for b in self.b], dim=0)
        # 水平拼接两个张量
        new_b = torch.cat([prompt_b_repeated, b_concatenated], dim=1)
        new_b = torch.index_select(new_b, dim=0, index=item2)
        new_b = self.fc3(new_b)
        new_b = torch.squeeze(new_b, dim=-1)

        if self.a_range is not None:
            new_a = self.a_range * torch.sigmoid(new_a)
            new_b = self.a_range * torch.sigmoid(new_b)
            new_theta = self.a_range * torch.sigmoid(new_theta)
        else:
            new_a = F.softplus(new_a)

        if torch.max(new_theta != new_theta) or torch.max(new_a != new_a) or torch.max(new_b != new_b):
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

        self.theta = nn.Embedding(self.user_num, self.latent_dim)
        self.transform_layer_stu = Transform_stu(self.pp_dim, self.s_ranges)

        self.a = nn.Embedding(self.item_num, latent_dim)
        self.prompt_a = nn.Embedding(self.item_num, self.pp_dim)

        self.b = nn.Embedding(self.item_num, 1)
        self.prompt_b = nn.Embedding(self.item_num, self.pp_dim)

        self.a_range = 1

        self.fc1 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)
        self.fc3 = nn.Linear(self.pp_dim + 1, 1)

    def forward(self, user, item):
        a = self.a(item)
        p_a = self.prompt_a(item)
        new_a = torch.cat([p_a, a], dim=1)
        new_a = self.fc1(new_a)

        theta = self.theta(user)
        new_theta = self.transform_layer_stu(theta)
        new_theta = self.fc2(new_theta)

        b = self.b(item)
        p_b = self.prompt_b(item)
        new_b = torch.cat([p_b, b], dim=1)
        new_b = self.fc3(new_b)
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

        self.theta = nn.Embedding(self.user_num, self.latent_dim)
        self.transform_layer_stu = Transform_stu(self.pp_dim, self.s_ranges)

        #self.a = nn.Embedding(self.item_num, latent_dim)
        self.generalize_layer_a = nn.Linear(self.pp_dim, self.latent_dim)
        self.prompt_a = nn.Embedding(self.item_num, self.pp_dim)

        #self.b = nn.Embedding(self.item_num, 1)
        self.generalize_layer_b = nn.Linear(self.pp_dim, 1)
        self.prompt_b = nn.Embedding(self.item_num, self.pp_dim)

        self.a_range = 1

        self.fc1 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.pp_dim + self.latent_dim, self.latent_dim)
        self.fc3 = nn.Linear(self.pp_dim + 1, 1)

    def forward(self, user, item):
        #a = self.a(item)
        p_a = self.prompt_a(item)
        a = self.generalize_layer_a(p_a)
        new_a = torch.cat([p_a, a], dim=1)
        new_a = self.fc1(new_a)

        theta = self.theta(user)
        new_theta = self.transform_layer_stu(theta)
        new_theta = self.fc2(new_theta)

        #b = self.b(item)
        p_b = self.prompt_b(item)
        b = self.generalize_layer_b(p_b)
        new_b = torch.cat([p_b, b], dim=1)
        new_b = self.fc3(new_b)
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
    def __init__(self, s_user_num, t_user_num, item_num, latent_dim, pp_dim, s_ranges, model_file,target_model_file,a_range=None):
        super(MIRT, self).__init__()
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
                auc, accuracy = self.Source_net_eval(self.s_irt_net,test_data, device=device)
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

    def Target_train(self, model , train_data, test_data=None, epoch=50, device="cpu", lr=0.001, silence=False, patience=5):
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
                user_id, item_id,y = batch_data
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


    def Source_net_eval(self, model , test_data, device="cpu"):
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

    def Target_net_eval(self, model , test_data, device="cpu"):
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

    def Transfer_parameters(self, target_net, s_ranges):
        # 加载模型、迁移参数
        self.s_irt_net.load_state_dict(torch.load(self.model_file))

        target_net.prompt_a.weight.data.copy_(
            self.s_irt_net.prompt_a.data)

        target_net.prompt_b.weight.data.copy_(
            self.s_irt_net.prompt_b.data)

        for i in range(len(s_ranges)):
            target_net.transform_layer_stu.s_stu_vectors[i].data.copy_(
                self.s_irt_net.s_stu_vectors[i].data)
            target_net.transform_layer_stu.s_stu_vectors[i].requires_grad = True

    def Transfer_parameters_temp(self,s_ranges):
        # 加载模型、迁移参数
        self.s_irt_net.load_state_dict(torch.load(self.model_file))

        self.t_irt_net.prompt_a.weight.data.copy_(
            self.s_irt_net.prompt_a.data)

        self.t_irt_net.prompt_b.weight.data.copy_(
            self.s_irt_net.prompt_b.data)

        self.t_irt_net.fc2.weight.data = self.s_irt_net.fc2.weight.data.clone()
        self.t_irt_net.fc2.bias.data = self.s_irt_net.fc2.bias.data.clone()

        for param in self.t_irt_net.fc2.parameters():
            param.requires_grad = False

        for i in range(len(s_ranges)):
            self.t_irt_net.transform_layer_stu.s_stu_vectors[i].data.copy_(
                self.s_irt_net.s_stu_vectors[i].data)
            self.t_irt_net.transform_layer_stu.s_stu_vectors[i].requires_grad = True


    def draw_student_distribution(self):
        # 加载模型
        source_theta = self.s_irt_net.theta.cpu()
        target_theta = self.t_irt_net.theta.weight.cpu()

        # 将两个向量垂直拼接
        combined_theta = torch.cat((source_theta, target_theta), dim=0)

        # 定义ID范围
        ranges = [range(1, 985), range(985, 1809), range(1809, 2204), range(2204, 3962)]

        # 从每个范围中随机抽取100个ID
        sample_indices = []
        range_labels = []
        for i, r in enumerate(ranges):
            sampled = np.random.choice(list(r), 100, replace=False)
            sample_indices.extend(sampled)
            range_labels.extend([i] * 100)  # 标记这些ID属于哪个范围

        # 从拼接后的向量中取出对应的向量样本
        sample_vectors = combined_theta[sample_indices]

        # 将样本转换为numpy数组以便使用sklearn
        sample_vectors_np = sample_vectors.detach().numpy()

        # PCA降维到2维
        pca = PCA(n_components=2)
        reduced_vectors = pca.fit_transform(sample_vectors_np)

        # K-means聚类
        kmeans = KMeans(n_clusters=4)
        labels = kmeans.fit_predict(reduced_vectors)

        # 定义颜色
        colors = ['red', 'blue', 'green', 'purple']

        # 绘制分布图，根据range_labels进行颜色划分
        plt.figure(figsize=(10, 7))
        for i in range(4):
            idx = np.where(np.array(range_labels) == i)
            plt.scatter(reduced_vectors[idx, 0], reduced_vectors[idx, 1], c=colors[i], label=f'Range {i + 1}')

        plt.title('PCA reduced vectors with K-means clustering')
        plt.xlabel('PCA component 1')
        plt.ylabel('PCA component 2')
        plt.legend()
        plt.show()

    def draw_student_distribution2(self):
        # 加载模型
        source_theta = self.s_irt_net.theta.cpu()

        # 定义ID范围
        ranges = [range(2263, 4021), range(0, 984), range(984, 1808), range(1808, 2263)]

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
        custom_labels = ['A-bin (Target)','B-bin (Source)', 'C-bin (Source)', 'D-bin (Source)']
        # 定义颜色
        colors = ['#4169E1', '#FF7F50', '#2E8B57', '#DAA520']

        # 设置字体为 Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'

        # 绘制分布图，根据range_labels进行颜色划分
        plt.figure(figsize=(7, 5))
        ax = plt.gca()
        ax.set_aspect(1,adjustable='datalim')

        for i in range(4):
            idx = np.where(np.array(range_labels) == i)
            plt.scatter(reduced_vectors[idx, 0], reduced_vectors[idx, 1], c=colors[i], label=custom_labels[i])

        plt.title('Student-Aspect Cross Domain (Without Prompt)', fontsize=20,fontweight='bold')
        plt.xlabel('Component 1', fontsize=20)
        plt.ylabel('Component 2', fontsize=20)
        # 将图例设置为2行2列，并放置在图形的外部
        plt.legend(fontsize=20, loc='upper right', ncol=2)
        plt.xticks(fontsize=18)  # 设置横坐标数字大小
        plt.yticks(fontsize=18)  # 设置纵坐标数字大小
        plt.show()

    # def draw_student_distribution3(self):
    #
    #     temp_vectors = torch.cat(
    #         [vector.repeat(r[1] - r[0] + 1, 1) for vector, r in zip(self.s_irt_net.s_stu_vectors, self.s_irt_net.s_ranges)], dim=0)
    #     all_theta = torch.cat([temp_vectors, self.s_irt_net.theta], dim=1)
    #     new_theta = self.s_irt_net.fc2(all_theta)
    #     new_theta = new_theta.cpu()
    #
    #     # 定义ID范围
    #     ranges = [range(1, 985), range(985, 1809), range(1809, 2264), range(2264, 4021)]
    #
    #     # 从每个范围中随机抽取100个ID
    #     sample_indices = []
    #     range_labels = []
    #     for i, r in enumerate(ranges):
    #         sampled = np.random.choice(list(r), 100, replace=False)
    #         sample_indices.extend(sampled)
    #         range_labels.extend([i] * 100)  # 标记这些ID属于哪个范围
    #
    #     # 取出对应的向量样本
    #     sample_vectors = new_theta[sample_indices]
    #
    #     # 将样本转换为numpy数组以便使用sklearn
    #     sample_vectors_np = sample_vectors.detach().numpy()
    #
    #     # PCA降维到2维
    #     pca = PCA(n_components=2)
    #     reduced_vectors = pca.fit_transform(sample_vectors_np)
    #
    #     # 自定义标签，可以根据实际需求修改这些标签
    #     custom_labels = ['schoolB', 'schoolC', 'schoolD', 'schoolA']
    #     # 定义颜色
    #     colors = ['red', 'blue', 'green', 'purple']
    #
    #     # 绘制分布图，根据range_labels进行颜色划分
    #     plt.figure(figsize=(10, 7))
    #     for i in range(4):
    #         idx = np.where(np.array(range_labels) == i)
    #         plt.scatter(reduced_vectors[idx, 0], reduced_vectors[idx, 1], c=colors[i], label=custom_labels[i])
    #
    #     plt.title('PCA reduced vectors with K-means clustering')
    #     plt.xlabel('PCA component 1')
    #     plt.ylabel('PCA component 2')
    #     plt.legend()
    #     plt.show()

    def draw_student_distribution3_1(self):
        temp_vectors = torch.cat(
            [vector.repeat(r[1] - r[0] + 1, 1) for vector, r in zip(self.s_irt_net.s_stu_vectors, self.s_irt_net.s_ranges)], dim=0)
        all_theta = torch.cat([temp_vectors, self.s_irt_net.theta], dim=1)
        new_theta = self.s_irt_net.fc2(all_theta)
        new_theta = new_theta.cpu()

        # 定义ID范围
        ranges = [range(2263, 4021), range(0, 984), range(984, 1808), range(1808, 2263)]

        # 从每个范围中随机抽取100个ID
        sample_indices = []
        range_labels = []
        for i, r in enumerate(ranges):
            sampled = np.random.choice(list(r), 100, replace=False)
            sample_indices.extend(sampled)
            range_labels.extend([i] * 100)  # 标记这些ID属于哪个范围

        # 取出对应的向量样本
        sample_vectors = new_theta[sample_indices]

        # 将样本转换为numpy数组以便使用sklearn
        sample_vectors_np = sample_vectors.detach().numpy()

        # 使用 t-SNE 降维到2维
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        reduced_vectors = tsne.fit_transform(sample_vectors_np)

        # 自定义标签，可以根据实际需求修改这些标签
        custom_labels = ['A-bin (Target)','B-bin (Source)', 'C-bin (Source)', 'D-bin (Source)']
        # 定义颜色
        colors = ['#4169E1', '#FF7F50', '#2E8B57', '#DAA520']

        # 设置字体为 Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'

        # 绘制分布图，根据range_labels进行颜色划分
        plt.figure(figsize=(7, 7))
        ax = plt.gca()
        ax.set_aspect(1,adjustable='datalim')

        for i in range(4):
            idx = np.where(np.array(range_labels) == i)
            plt.scatter(reduced_vectors[idx, 0], reduced_vectors[idx, 1], c=colors[i], label=custom_labels[i])

        plt.title('Student-Aspect Cross Domain (With Prompt)', fontsize=20,fontweight='bold')
        plt.xlabel('Component 1', fontsize=18)
        plt.ylabel('Component 2', fontsize=18)
        plt.legend(fontsize=20, loc='upper right', ncol=2)
        plt.xticks(fontsize=18)  # 设置横坐标数字大小
        plt.yticks(fontsize=18)  # 设置纵坐标数字大小
        plt.legend(fontsize=18)
        plt.show()

    def draw_student_distribution4(self):

        temp_vectors = torch.cat(
            [vector.repeat(r[1] - r[0] + 1, 1) for vector, r in zip(self.s_irt_net.s_stu_vectors, self.s_irt_net.s_ranges)], dim=0)
        all_theta = torch.cat([temp_vectors, self.s_irt_net.theta], dim=1)
        new_theta_source = self.s_irt_net.fc2(all_theta)
        new_theta_source = new_theta_source.cpu()

        # self.theta = nn.Embedding(self.user_num, self.latent_dim)
        # self.transform_layer_stu = Transform_stu(self.pp_dim, self.s_ranges)
        #
        # theta = self.t_irt_net.theta(user)'
        theta = self.t_irt_net.theta.weight.data
        new_theta_target = self.t_irt_net.transform_layer_stu(theta)
        new_theta_target = self.t_irt_net.fc2(new_theta_target)
        new_theta_target = new_theta_target.cpu()

        # 将两个向量垂直拼接
        combined_theta = torch.cat((new_theta_source, new_theta_target), dim=0)

        # 定义ID范围
        ranges = [range(1, 985), range(985, 1809), range(1809, 2204), range(2204, 3962)]

        # 从每个范围中随机抽取100个ID
        sample_indices = []
        range_labels = []
        for i, r in enumerate(ranges):
            sampled = np.random.choice(list(r), 100, replace=False)
            sample_indices.extend(sampled)
            range_labels.extend([i] * 100)  # 标记这些ID属于哪个范围

        # 从拼接后的向量中取出对应的向量样本
        sample_vectors = combined_theta[sample_indices]

        # 将样本转换为numpy数组以便使用sklearn
        sample_vectors_np = sample_vectors.detach().numpy()

        # PCA降维到2维
        pca = PCA(n_components=2)
        reduced_vectors = pca.fit_transform(sample_vectors_np)

        # K-means聚类
        kmeans = KMeans(n_clusters=4)
        labels = kmeans.fit_predict(reduced_vectors)

        # 定义颜色
        colors = ['red', 'blue', 'green', 'purple']

        # 绘制分布图，根据range_labels进行颜色划分
        plt.figure(figsize=(10, 7))
        for i in range(4):
            idx = np.where(np.array(range_labels) == i)
            plt.scatter(reduced_vectors[idx, 0], reduced_vectors[idx, 1], c=colors[i], label=f'Range {i + 1}')

        plt.title('PCA reduced vectors with K-means clustering')
        plt.xlabel('PCA component 1')
        plt.ylabel('PCA component 2')
        plt.legend()
        plt.show()

    def draw_student_distribution5(self):

        temp_vectors = torch.cat(
            [vector.repeat(r[1] - r[0] + 1, 1) for vector, r in
             zip(self.s_irt_net.s_stu_vectors, self.s_irt_net.s_ranges)], dim=0)
        all_theta = torch.cat([temp_vectors, self.s_irt_net.theta], dim=1)
        all_theta = all_theta.cpu()

        # self.theta = nn.Embedding(self.user_num, self.latent_dim)
        # self.transform_layer_stu = Transform_stu(self.pp_dim, self.s_ranges)
        #
        # theta = self.t_irt_net.theta(user)'
        theta = self.t_irt_net.theta.weight.data
        new_theta_target = self.t_irt_net.transform_layer_stu(theta)
        new_theta_target = new_theta_target.cpu()

        # 将两个向量垂直拼接
        combined_theta = torch.cat((all_theta, new_theta_target), dim=0)

        # 定义ID范围
        ranges = [range(1, 985), range(985, 1809), range(1809, 2204), range(2204, 3962)]

        # 从每个范围中随机抽取100个ID
        sample_indices = []
        range_labels = []
        for i, r in enumerate(ranges):
            sampled = np.random.choice(list(r), 100, replace=False)
            sample_indices.extend(sampled)
            range_labels.extend([i] * 100)  # 标记这些ID属于哪个范围

        # 从拼接后的向量中取出对应的向量样本
        sample_vectors = combined_theta[sample_indices]

        # 将样本转换为numpy数组以便使用sklearn
        sample_vectors_np = sample_vectors.detach().numpy()

        # PCA降维到2维
        pca = PCA(n_components=2)
        reduced_vectors = pca.fit_transform(sample_vectors_np)

        # K-means聚类
        kmeans = KMeans(n_clusters=4)
        labels = kmeans.fit_predict(reduced_vectors)

        # 定义颜色
        colors = ['red', 'blue', 'green', 'purple']

        # 绘制分布图，根据range_labels进行颜色划分
        plt.figure(figsize=(10, 7))
        for i in range(4):
            idx = np.where(np.array(range_labels) == i)
            plt.scatter(reduced_vectors[idx, 0], reduced_vectors[idx, 1], c=colors[i], label=f'Range {i + 1}')

        plt.title('PCA reduced vectors with K-means clustering')
        plt.xlabel('PCA component 1')
        plt.ylabel('PCA component 2')
        plt.legend()
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