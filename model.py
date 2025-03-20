import torch
from torch import nn


class Squash(nn.Module):
    #input：[batch_size, n_capsules, n_features]
    #output：[batch_size, n_capsules, n_features]
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, s: torch.Tensor):  # s: [batch_size, n_capsules, n_features]
        s2 = (s ** 2).sum(dim=-1, keepdims=True)
        return (s2 / (1 + s2)) * (s / torch.sqrt(s2 + self.epsilon))


#胶囊网络的路由机制
class Router(nn.Module):
    #input：[batch_size, in_caps, in_d]（低层胶囊的输出）
    #output：[batch_size, out_caps, out_d]（高层胶囊的输出）
    def __init__(self, in_caps: int, out_caps: int, in_d: int, out_d: int, iterations: int):  # int_d: 前一层胶囊的特征数目
        super().__init__()
        self.in_caps = in_caps  # 胶囊数目
        self.out_caps = out_caps
        self.iterations = iterations    #轮次
        self.softmax = nn.Softmax(dim=1)
        self.squash = Squash()

        # maps each capsule in the lower layer to each capsule in this layer
        self.weight = nn.Parameter(torch.randn(in_caps, out_caps, in_d, out_d), requires_grad=True)

    def forward(self, u: torch.Tensor):  # 低层胶囊的输入
        """
        input(s) shape: [batch_size, n_capsules, n_features]
        output shape: [batch_size, n_capsules, n_features]
        """

        u_hat = torch.einsum('ijnm,bin->bijm', self.weight, u)  # 计算预测输出，每个输入向量对每个类别有一个预测
        '''
        对于每个 batch_size 中的样本，计算每个输入胶囊 i 对应的所有输出胶囊 j 的预测输出 u_hat(权值矩阵乘输入矩阵)
        i 对应 in_caps，表示输入胶囊的索引。
        j 对应 out_caps，表示输出胶囊的索引。，即类别数 2
        n 对应 in_d，表示输入胶囊的特征维度。
        m 对应 out_d，表示输出胶囊的特征维度。
        b 对应 batch_size，表示每个样本的大小。
        '''
        b = u.new_zeros(u.shape[0], self.in_caps, self.out_caps)    # 初始化信号强度
        v = None    # 初始化胶囊输出
        for i in range(self.iterations):
            c = self.softmax(b) # 动态路由中的加权系数
            s = torch.einsum('bij,bijm->bjm', c, u_hat) # 计算加权和
            v = self.squash(s)  ## 激活并返回
            a = torch.einsum('bjm,bijm->bij', v, u_hat) # 更新信号强度
            b = b + a   #更新
        return v    # 返回胶囊网络的输出


class cross_attention(nn.Module):
    def __init__(self):
        super(cross_attention, self).__init__()
        #生成Q,K,V

        self.linearK = nn.Sequential(nn.Linear(1024, 531))
        self.linearQ = nn.Sequential(nn.Linear(531, 531))
        self.linearV = nn.Sequential(nn.Linear(1024, 1024))

        ##softmax：用于生成注意力系数的激活函数。
        self.softmax = nn.Softmax()

        self.linear_mid = nn.Linear(1024, 1024)
        self.linear_final = nn.Linear(1024, 1024)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.2)
        self.cnn = nn.Sequential(
            nn.Conv1d(
            in_channels= 1024,
            out_channels= 512,
            kernel_size=3,
            stride=1,
            padding=1,
        ),
            nn.BatchNorm1d(512),
            nn.ELU(),  # activation
            nn.Conv1d(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm1d(256),
            nn.ELU(),  # activation
        )

        self.capsule_layer = Router(in_caps=256, out_caps=2,
                                    in_d=25, out_d=8,iterations=2)


        self.fc = nn.Sequential(nn.Linear(16, 64),
                                nn.ELU(),
                                nn.Dropout(0.3),
                                nn.Linear(64, 2))



    def forward(self, x_t5, x_aaindex):
        '''
        :x_t5:Prot_T5，维度 (batch_size, length = 25, feature_dim = 1024)
        :x_aaindex:AAindex (batch_size, length = 25, feature_dim = 531)
        '''
        Q, K, V = x_aaindex, x_t5, x_t5
        #计算K，Q
        #线性层逐行操作，不会跨行（即不同氨基酸）共享权重或进行加权。因此，一个氨基酸的特征向量（531 维）不会直接受到其他氨基酸特征向量的影响
        K = self.linearK(K)
        Q = self.linearQ(Q)
        V = self.linearV(V)
        # einsum用来执行矩阵乘法、张量相乘、转置、合并等操作。
        # "bij,bkj->bik"：i 和 k 保留，而 j 是求和维度。即，对于每个节点 i，计算其与邻居 k 的相似度（或者注意力权重）
        # 具体维度变化情况，（batch_size, 25, 531），（batch_size, 25, 531）->batch_size, 25, 25）
        '''这一步使用爱因斯坦求和约定 (torch.einsum) 来计算注意力权重矩阵 attn。'''
        attn = torch.einsum('bij,bkj->bik', Q, K)

        # 计算每个节点的注意力权重之和，用于后续归一化。"bij->bi"：对维度 j 求和
        # (batch_size, 25, 25）->(batch_size, 25）
        attn_sum = torch.einsum('bij->bi', attn)
        # 行归一化，使得每个节点的注意力系数总和为 1。bij,bi->bij 在特定维度上进行逐元素相乘
        # (batch_size, 25, 25）,(batch_size, 25）->(batch_size, 25, 25）
        attn = torch.einsum('bij,bi->bij', attn, 1 / attn_sum)
        # 记录注意力权重
        #attns.append(attn.cpu())
        # 使用注意力权重更新特征，"bij,bjd->bid"：使用注意力权重对邻居特征求加权和
        # (batch_size, 25, 25）,(batch_size, 25, 1024）->(batch_size, 25, 1024）
        x_t5 = torch.einsum('bij,bjd->bid', attn, V)
        # 将更新后的节点特征 x 通过 linear_mid，并应用 eLU 激活函数
        x_t5 = self.dropout(self.elu(self.linear_mid(x_t5)))

        x_t5 = x_t5 + V

        att_out = self.linear_final(x_t5)

        att_out = att_out.permute(0, 2, 1)

        cnn_out = self.cnn(att_out)
        cap_out = self.capsule_layer(cnn_out)

        #展开
        view = cap_out.view(cap_out.size(0),-1)  # 将网络展开(batch_size,2，8) -> (batchsize,2*8)

        out = self.fc(view)

        return out, attn