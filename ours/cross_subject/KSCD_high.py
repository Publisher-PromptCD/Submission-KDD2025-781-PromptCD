import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score
from torch.utils.data import DataLoader, TensorDataset, random_split

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

class Source_Net(nn.Module):
    def __init__(self, knowledge_n, exer_n, student_n, low_dim, pp_dim, s_ranges):
        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.stu_n = student_n
        self.low_dim = low_dim
        self.pp_dim = pp_dim
        self.s_ranges = s_ranges
        self.net1 = knowledge_n # 表示学生在特定知识点掌握的表征维度
        self.prednet_input_len = self.knowledge_n
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Source_Net, self).__init__()

        self.student_emb = nn.ParameterList([nn.Parameter(torch.randn(self.stu_n, self.low_dim))
                                       for _ in range(len(s_ranges))])
        # 对每个参数进行 Xavier 均匀初始化
        for student_emb in self.student_emb:
            nn.init.xavier_uniform_(student_emb)
        self.prompt_stu = nn.Parameter(torch.randn(self.stu_n, self.pp_dim))
        nn.init.xavier_uniform_(self.prompt_stu)

        self.knowledge_emb = nn.Embedding(self.knowledge_n, self.low_dim)  # 知识点矩阵的低维表示

        self.k_difficulty = nn.Parameter(torch.rand((self.exer_n, self.low_dim)))
        nn.init.xavier_uniform_(self.k_difficulty)
        self.s_exer_vectors = nn.ParameterList([nn.Parameter(torch.rand(self.pp_dim)) for _ in range(len(s_ranges))])

        self.k_index = torch.LongTensor(list(range(self.knowledge_n)))

        self.prednet_full1 = nn.Linear(self.knowledge_n + self.low_dim, self.net1, bias=False)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.knowledge_n + self.low_dim, self.net1, bias=False)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(1 * self.net1, 1)
        self.layer1 = nn.Linear(self.low_dim, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

        self.fc1 = nn.Linear(self.pp_dim + self.knowledge_n, self.knowledge_n)
        self.fc2 = nn.Linear(self.pp_dim + self.knowledge_n, self.knowledge_n)

    def forward(self, stu_id, exer_id, kn_emb):

        knowledge_low_emb = self.knowledge_emb(self.k_index)
        # step2.
        # get batch student data
        #-----------------------------------------------------
        # 将参数 prompt_theta 重复 n 次
        prompt_stu_repeated = self.prompt_stu.repeat(len(self.s_ranges), 1)
        # 将列表中的每个元素垂直拼接起来
        stu_concatenated = torch.cat([stu for stu in self.student_emb], dim=0)
        # 从两个参数中取出相应的向量
        prompt_stu = torch.index_select(prompt_stu_repeated, dim=0, index=stu_id)
        old_stu = torch.index_select(stu_concatenated, dim=0, index=stu_id)
        # 与知识点矩阵做乘法
        old_stu = torch.sigmoid(torch.mm(old_stu, knowledge_low_emb.T))
        # 加上prompt
        new_stu = torch.cat([prompt_stu, old_stu], dim=1)
        # 过全连接层再映射回去
        batch_stu_emb = torch.sigmoid(self.fc1(new_stu))
        #-----------------------------------------------------

        batch_stu_vector = batch_stu_emb.repeat(1, self.knowledge_n).reshape(batch_stu_emb.shape[0], self.knowledge_n, batch_stu_emb.shape[1])

        # get batch exercise data
        #------------------------------------------------------------------
        # 制作每个域的prompt矩阵
        temp_vectors = torch.cat(
            [vector.repeat(r[1] - r[0] + 1, 1) for vector, r in zip(self.s_exer_vectors, self.s_ranges)], dim=0)
        # 取出相应的向量
        prompt_exer = torch.sigmoid(torch.index_select(temp_vectors, dim=0, index=exer_id))
        old_exer = torch.sigmoid(torch.index_select(self.k_difficulty, dim=0, index=exer_id))
        # 乘法
        old_exer = torch.sigmoid(torch.mm(old_exer, knowledge_low_emb.T))
        # 加上prompt
        new_exer = torch.cat([prompt_exer, old_exer], dim=1)
        # 过全连接层再映射回去
        batch_exer_emb = torch.sigmoid(self.fc2(new_exer))

        #------------------------------------------------------------------
        batch_exer_vector = batch_exer_emb.repeat(1, self.knowledge_n).reshape(batch_exer_emb.shape[0], self.knowledge_n, batch_exer_emb.shape[1])
        #         print("batch_exer_vector:", batch_exer_vector.shape)

        # get batch knowledge concept data
        kn_vector = knowledge_low_emb.repeat(batch_stu_emb.shape[0], 1).reshape(batch_stu_emb.shape[0],
                                                                                knowledge_low_emb.shape[0],
                                                                                knowledge_low_emb.shape[1])

        # CD
        preference = torch.sigmoid(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2)))
        diff = torch.sigmoid(self.prednet_full2(torch.cat((batch_exer_vector, kn_vector), dim=2)))
        o = torch.sigmoid(self.prednet_full3(preference - diff))

        sum_out = torch.sum(o * kn_emb.unsqueeze(2), dim=1)
        count_of_concept = torch.sum(kn_emb, dim=1).unsqueeze(1)
        output = sum_out / count_of_concept
        return output

class Target_Net(nn.Module):
    def __init__(self, knowledge_n, exer_n, student_n, low_dim, pp_dim, s_ranges):
        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.stu_n = student_n
        self.low_dim = low_dim
        self.pp_dim = pp_dim
        self.s_ranges = s_ranges
        self.net1 = knowledge_n # 表示学生在特定知识点掌握的表征维度
        self.prednet_input_len = self.knowledge_n
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Target_Net, self).__init__()

        self.student_emb = nn.Embedding(self.stu_n, self.low_dim)
        self.prompt_stu = nn.Embedding(self.stu_n, self.pp_dim)

        self.knowledge_emb = nn.Embedding(self.knowledge_n, self.low_dim)  # 知识点矩阵的低维表示

        self.k_difficulty = nn.Embedding(self.exer_n, self.low_dim)
        self.transform_layer_exr = Transform_Exr(self.pp_dim, self.s_ranges)

        self.k_index = torch.LongTensor(list(range(self.knowledge_n)))

        self.prednet_full1 = nn.Linear(self.knowledge_n + self.low_dim, self.net1, bias=False)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.knowledge_n + self.low_dim, self.net1, bias=False)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(1 * self.net1, 1)
        self.layer1 = nn.Linear(self.low_dim, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

        self.fc1 = nn.Linear(self.pp_dim + self.knowledge_n, self.knowledge_n)
        self.fc2 = nn.Linear(self.pp_dim + self.knowledge_n, self.knowledge_n)

    def forward(self, stu_id, exer_id, kn_emb):
        knowledge_low_emb = self.knowledge_emb(self.k_index)
        # step2.
        # get batch student data
        #-----------------------------------------------------
        # 从两个参数中取出相应的向量
        prompt_stu = self.prompt_stu(stu_id)
        old_stu = self.student_emb(stu_id)
        # 相乘
        old_stu = torch.sigmoid(torch.mm(old_stu, knowledge_low_emb.T))
        # 拼接
        new_stu = torch.cat([prompt_stu,old_stu],dim=1)
        # 过全连接层再映射回去
        batch_stu_emb = torch.sigmoid(self.fc1(new_stu))
        #-----------------------------------------------------

        batch_stu_vector = batch_stu_emb.repeat(1, self.knowledge_n).reshape(batch_stu_emb.shape[0], self.knowledge_n, batch_stu_emb.shape[1])

        # get batch exercise data
        #------------------------------------------------------------------
        # 从参数中取出相应的向量
        old_exer = self.k_difficulty(exer_id)
        # 相乘
        old_exer = torch.sigmoid(torch.mm(old_exer, knowledge_low_emb.T))
        # 拼接
        new_exer = self.transform_layer_exr(old_exer)
        # 过全连接层再映射回去
        batch_exer_emb = torch.sigmoid(self.fc2(new_exer))

        #------------------------------------------------------------------
        batch_exer_vector = batch_exer_emb.repeat(1, self.knowledge_n).reshape(batch_exer_emb.shape[0], self.knowledge_n, batch_exer_emb.shape[1])
        #         print("batch_exer_vector:", batch_exer_vector.shape)

        # get batch knowledge concept data
        kn_vector = knowledge_low_emb.repeat(batch_stu_emb.shape[0], 1).reshape(batch_stu_emb.shape[0],
                                                                                knowledge_low_emb.shape[0],
                                                                                knowledge_low_emb.shape[1])

        # CD
        preference = torch.sigmoid(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2)))
        diff = torch.sigmoid(self.prednet_full2(torch.cat((batch_exer_vector, kn_vector), dim=2)))
        o = torch.sigmoid(self.prednet_full3(preference - diff))

        sum_out = torch.sum(o * kn_emb.unsqueeze(2), dim=1)
        count_of_concept = torch.sum(kn_emb, dim=1).unsqueeze(1)
        output = sum_out / count_of_concept
        return output

class Target_Net2(nn.Module):
    def __init__(self, knowledge_n, exer_n, student_n, low_dim, pp_dim, s_ranges):
        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.stu_n = student_n
        self.low_dim = low_dim
        self.pp_dim = pp_dim
        self.s_ranges = s_ranges
        self.net1 = knowledge_n # 表示学生在特定知识点掌握的表征维度
        self.prednet_input_len = self.knowledge_n
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Target_Net2, self).__init__()

        #self.student_emb = nn.Embedding(self.stu_n, self.low_dim)
        self.prompt_stu = nn.Embedding(self.stu_n, self.pp_dim)
        self.generalize_layer_stu = nn.Linear(self.pp_dim, self.low_dim)

        self.knowledge_emb = nn.Embedding(self.knowledge_n, self.low_dim)  # 知识点矩阵的低维表示

        self.k_difficulty = nn.Embedding(self.exer_n, self.low_dim)
        self.transform_layer_exr = Transform_Exr(self.pp_dim, self.s_ranges)

        self.k_index = torch.LongTensor(list(range(self.knowledge_n)))

        self.prednet_full1 = nn.Linear(self.knowledge_n + self.low_dim, self.net1, bias=False)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.knowledge_n + self.low_dim, self.net1, bias=False)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(1 * self.net1, 1)
        self.layer1 = nn.Linear(self.low_dim, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

        self.fc1 = nn.Linear(self.pp_dim + self.knowledge_n, self.knowledge_n)
        self.fc2 = nn.Linear(self.pp_dim + self.knowledge_n, self.knowledge_n)

    def forward(self, stu_id, exer_id, kn_emb):
        knowledge_low_emb = self.knowledge_emb(self.k_index)
        # step2.
        # get batch student data
        #-----------------------------------------------------
        # 从两个参数中取出相应的向量
        prompt_stu = self.prompt_stu(stu_id)
        #old_stu = self.student_emb(stu_id)
        old_stu =self.generalize_layer_stu(prompt_stu)
        # 相乘
        old_stu = torch.sigmoid(torch.mm(old_stu, knowledge_low_emb.T))
        # 拼接
        new_stu = torch.cat([prompt_stu,old_stu],dim=1)
        # 过全连接层再映射回去
        batch_stu_emb = torch.sigmoid(self.fc1(new_stu))
        #-----------------------------------------------------

        batch_stu_vector = batch_stu_emb.repeat(1, self.knowledge_n).reshape(batch_stu_emb.shape[0], self.knowledge_n, batch_stu_emb.shape[1])

        # get batch exercise data
        #------------------------------------------------------------------
        # 从参数中取出相应的向量
        old_exer = self.k_difficulty(exer_id)
        # 相乘
        old_exer = torch.sigmoid(torch.mm(old_exer, knowledge_low_emb.T))
        # 拼接
        new_exer = self.transform_layer_exr(old_exer)
        # 过全连接层再映射回去
        batch_exer_emb = torch.sigmoid(self.fc2(new_exer))

        #------------------------------------------------------------------
        batch_exer_vector = batch_exer_emb.repeat(1, self.knowledge_n).reshape(batch_exer_emb.shape[0], self.knowledge_n, batch_exer_emb.shape[1])
        #         print("batch_exer_vector:", batch_exer_vector.shape)

        # get batch knowledge concept data
        kn_vector = knowledge_low_emb.repeat(batch_stu_emb.shape[0], 1).reshape(batch_stu_emb.shape[0],
                                                                                knowledge_low_emb.shape[0],
                                                                                knowledge_low_emb.shape[1])

        # CD
        preference = torch.sigmoid(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2)))
        diff = torch.sigmoid(self.prednet_full2(torch.cat((batch_exer_vector, kn_vector), dim=2)))
        o = torch.sigmoid(self.prednet_full3(preference - diff))

        sum_out = torch.sum(o * kn_emb.unsqueeze(2), dim=1)
        count_of_concept = torch.sum(kn_emb, dim=1).unsqueeze(1)
        output = sum_out / count_of_concept
        return output

class Net(nn.Module):
    def __init__(self, knowledge_n, exer_n, student_n, low_dim, pp_dim, s_ranges):
        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.stu_n = student_n
        self.low_dim = low_dim
        self.pp_dim = pp_dim
        self.s_ranges = s_ranges
        self.net1 = knowledge_n  # 表示学生在特定知识点掌握的表征维度
        self.prednet_input_len = self.knowledge_n
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net, self).__init__()

        # network structure
        self.student_emb = nn.Embedding(self.stu_n, self.low_dim)  # 学生的低维表示
        self.knowledge_emb = nn.Embedding(self.knowledge_n, self.low_dim)  # 知识点矩阵的低维表示
        self.k_difficulty = nn.Embedding(self.exer_n, self.low_dim)  # 习题的低维表示

        self.k_index = torch.LongTensor(list(range(self.knowledge_n)))

        self.prednet_full1 = nn.Linear(self.knowledge_n + self.low_dim, self.net1, bias=False)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.knowledge_n + self.low_dim, self.net1, bias=False)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(1 * self.net1, 1)
        self.layer1 = nn.Linear(self.low_dim, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb):
        """
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors
        :return: FloatTensor, the probabilities of answering correctly
        """
        # before prednet
        knowledge_low_emb = self.knowledge_emb(self.k_index) # 32 123 123
        # step2.
        # get batch student data
        batch_stu_emb = self.student_emb(stu_id)
        batch_stu_emb = torch.sigmoid(torch.mm(batch_stu_emb, knowledge_low_emb.T))

        batch_stu_vector = batch_stu_emb.repeat(1, self.knowledge_n).reshape(batch_stu_emb.shape[0], self.knowledge_n, batch_stu_emb.shape[1])

        # get batch exercise data
        batch_exer_emb = self.k_difficulty(exer_id)  # 32 123
        #         e_discrimination = torch.sigmoid(self.layer1(batch_exer_emb))
        batch_exer_emb = torch.sigmoid(torch.mm(batch_exer_emb, knowledge_low_emb.T))

        batch_exer_vector = batch_exer_emb.repeat(1, self.knowledge_n).reshape(batch_exer_emb.shape[0], self.knowledge_n, batch_exer_emb.shape[1])
        #         print("batch_exer_vector:", batch_exer_vector.shape)

        # get batch knowledge concept data
        kn_vector = knowledge_low_emb.repeat(batch_stu_emb.shape[0], 1).reshape(batch_stu_emb.shape[0],
                                                                                knowledge_low_emb.shape[0],
                                                                                knowledge_low_emb.shape[1])

        # CD
        preference = torch.sigmoid(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2)))
        diff = torch.sigmoid(self.prednet_full2(torch.cat((batch_exer_vector, kn_vector), dim=2)))
        o = torch.sigmoid(self.prednet_full3(preference - diff))

        sum_out = torch.sum(o * kn_emb.unsqueeze(2), dim=1)
        count_of_concept = torch.sum(kn_emb, dim=1).unsqueeze(1)
        output = sum_out / count_of_concept
        return output

class KSCD:
    '''Neural Cognitive Diagnosis Model'''

    def __init__(self, source_knowledge_n, target_knowledge_n, source_exer_n, target_exer_n,
                 student_n, low_dim, pp_dim,s_ranges,model_file, target_model_file):

        super(KSCD, self).__init__()
        self.pp_dim = pp_dim
        self.latent_dim = target_knowledge_n
        self.model_file = model_file
        self.target_model_file = target_model_file
        self.kscd_s_net = Source_Net(source_knowledge_n, source_exer_n, student_n, low_dim, pp_dim, s_ranges)
        self.kscd_t_net = Target_Net(target_knowledge_n, target_exer_n, student_n, low_dim, pp_dim, s_ranges)
        self.kscd_t_net2 = Target_Net(target_knowledge_n, target_exer_n, student_n, low_dim, pp_dim, s_ranges)

    def Source_train(self, train_data, test_data=None, max_epochs=50, early_stopping_patience=2, device="cpu",
                     lr=0.001, silence=False):
        self.kscd_s_net = self.kscd_s_net.to(device)
        self.kscd_s_net.k_index = self.kscd_s_net.k_index.to(device)
        self.kscd_s_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.kscd_s_net.parameters(), lr=lr)

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

                pred: torch.Tensor = self.kscd_s_net(user_id, item_id, knowledge_emb)
                loss = loss_function(pred.squeeze(1), y)

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
                # 保存最佳模型
                if e > 0.001:
                    best_auc = auc
                    consecutive_no_improvement = 0

                    # 保存模型
                    torch.save(self.kscd_s_net.state_dict(), self.model_file)
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

    def Target_train(self, model, train_data, test_data=None, epoch=50, device="cpu", lr=0.001, silence=False, patience=2):
        kscd_t_net = model.to(device)
        kscd_t_net.k_index = kscd_t_net.k_index.to(device)
        kscd_t_net.train()

        loss_function = nn.BCELoss()
        optimizer = optim.Adam(kscd_t_net.parameters(), lr=lr)

        best_auc = 0.0  # 初始化为较低的值
        best_metrics = None  # 初始化为None
        early_stop_counter = 0  # 早停计数器

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
                pred: torch.Tensor = kscd_t_net(user_id, item_id, knowledge_emb)
                loss = loss_function(pred.squeeze(1), y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            average_loss = np.mean(epoch_losses)
            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(average_loss)))

            if test_data is not None:
                auc, accuracy, rmse, f1 = self.Target_net_eval(kscd_t_net, test_data, device=device)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (epoch_i, auc, accuracy))

                e = auc - best_auc
                # Update best metrics if current metrics are better
                if e > 0.0001:
                    best_auc = auc
                    best_metrics = (auc, accuracy, rmse, f1)
                    early_stop_counter = 0  # 重置早停计数器
                    torch.save(kscd_t_net.state_dict(), self.target_model_file)
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
        self.kscd_s_net = self.kscd_s_net.to(device)
        self.kscd_s_net.k_index = self.kscd_s_net.k_index.to(device)
        self.kscd_s_net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred: torch.Tensor = self.kscd_s_net(user_id, item_id, knowledge_emb)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def Target_net_eval(self, model, test_data, device="cpu"):
        kscd_t_net = model.to(device)
        kscd_t_net.k_index = kscd_t_net.k_index.to(device)
        kscd_t_net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred: torch.Tensor = kscd_t_net(user_id, item_id, knowledge_emb)
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
        self.kscd_s_net.load_state_dict(torch.load(self.model_file))

        model.prompt_stu.weight.data.copy_(
            self.kscd_s_net.prompt_stu.data)

        for i in range(len(s_ranges)):
            model.transform_layer_exr.s_exer_vectors[i].data.copy_(
                self.kscd_s_net.s_exer_vectors[i].data)
            model.transform_layer_exr.s_exer_vectors[i].requires_grad = True

    def Target_test(self, model, test_data, device="cpu"):
        kscd_t_net = model.to(device)
        kscd_t_net.k_index = kscd_t_net.k_index.to(device)
        kscd_t_net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred: torch.Tensor = kscd_t_net(user_id, item_id, knowledge_emb)
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

