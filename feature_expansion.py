import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch中的函数式API
import networkx as nx  # 导入NetworkX库，用于图结构操作
from torch_geometric.nn.conv import MessagePassing  # 导入PyTorch Geometric中的消息传递类
from torch_scatter import scatter_add  # 导入torch_scatter中的scatter_add函数
from torch_geometric.utils import degree  # 导入计算度的工具
from torch_geometric.utils import remove_self_loops, add_self_loops  # 导入移除和添加自环边的工具

class FeatureExpander(MessagePassing):
    r"""扩展特征。

    参数:
        degree (bool): 是否使用度特征。
        onehot_maxdeg (int): 是否使用one-hot度特征，设置最大度。0表示禁用。
        AK (int): 是否使用a^kx特征。0表示禁用。
        centrality (bool): 是否使用中心性特征。
        remove_edges (strings): 是否移除边，部分或全部。
        edge_noises_add (float): 添加随机边的比例。
        edge_noises_delete (float): 移除随机边的比例。
        group_degree (int): 按度分组节点创建超级节点，0表示禁用。
    """

    def __init__(self, degree=True, onehot_maxdeg=0, AK=1,
                 centrality=False, remove_edges="none",
                 edge_noises_add=0, edge_noises_delete=0, group_degree=0):
        super(FeatureExpander, self).__init__('add')  # 初始化MessagePassing父类
        self.degree = degree  # 是否使用度特征
        self.onehot_maxdeg = onehot_maxdeg  # 是否使用one-hot度特征，设置最大度
        self.AK = AK  # 是否使用a^kx特征
        self.centrality = centrality  # 是否使用中心性特征
        self.remove_edges = remove_edges  # 是否移除边，部分或全部
        self.edge_noises_add = edge_noises_add  # 添加随机边的比例
        self.edge_noises_delete = edge_noises_delete  # 移除随机边的比例
        self.group_degree = group_degree  # 按度分组节点创建超级节点
        assert remove_edges in ["none", "nonself", "all"], remove_edges  # 确保remove_edges参数有效

        self.edge_norm_diag = 1e-8  # 边规范化用于将A对角线设置为此值

    def transform(self, data):
        if data.x is None:  # 如果节点特征为空
            data.x = torch.ones([data.num_nodes, 1], dtype=torch.float)  # 初始化为全1

        # 在计算其他之前先添加边噪声
        if self.edge_noises_delete > 0:  # 如果设置了移除边的比例
            num_edges_new = data.num_edges - int(data.num_edges * self.edge_noises_delete)  # 计算移除后的边数
            idxs = torch.randperm(data.num_edges)[:num_edges_new]  # 随机选择保留的边索引
            data.edge_index = data.edge_index[:, idxs]  # 更新边索引
        if self.edge_noises_add > 0:  # 如果设置了添加边的比例
            num_new_edges = int(data.num_edges * self.edge_noises_add)  # 计算新增边的数量
            idx = torch.LongTensor(num_new_edges * 2).random_(0, data.num_nodes)  # 生成新增边的节点索引
            new_edges = idx.reshape(2, -1)  # 重新塑造为边索引
            data.edge_index = torch.cat([data.edge_index, new_edges], 1)  # 合并原有边和新增边

        deg, deg_onehot = self.compute_degree(data.edge_index, data.num_nodes)  # 计算节点度及其one-hot编码
        akx = self.compute_akx(data.num_nodes, data.x, data.edge_index)  # 计算a^kx特征
        cent = self.compute_centrality(data)  # 计算中心性特征
        data.x = torch.cat([data.x, deg, deg_onehot, akx, cent], -1)  # 合并所有特征

        if self.remove_edges != "none":  # 如果需要移除边
            if self.remove_edges == "all":  # 如果移除所有边
                self_edge = None  # 设置边为空
            else:  # 仅保留自环边
                self_edge = torch.tensor(range(data.num_nodes)).view((1, -1))  # 创建自环边索引
                self_edge = torch.cat([self_edge, self_edge], 0)  # 拼接自环边索引
            data.edge_index = self_edge  # 更新边索引

        # 通过基于度的分组减少节点
        if self.group_degree > 0:  # 如果设置了分组度
            assert self.remove_edges == "all", "remove all edges"  # 确保已移除所有边
            x_base = data.x  # 初始化基础特征
            deg_base = deg.view(-1)  # 展平度特征
            super_nodes = []  # 初始化超级节点列表
            for k in range(1, self.group_degree + 1):  # 遍历每个度
                eq_idx = deg_base == k  # 找到度等于k的节点
                gt_idx = deg_base > k  # 找到度大于k的节点
                x_to_group = x_base[eq_idx]  # 选取度等于k的节点特征
                x_base = x_base[gt_idx]  # 更新基础特征为度大于k的节点特征
                deg_base = deg_base[gt_idx]  # 更新基础度为度大于k的节点度
                group_size = torch.zeros([1, 1]) + x_to_group.size(0)  # 计算分组大小
                if x_to_group.size(0) == 0:  # 如果没有度等于k的节点
                    super_nodes.append(torch.cat([group_size, data.x[:1] * 0], -1))  # 添加空节点
                else:
                    super_nodes.append(torch.cat([group_size, x_to_group.mean(0, keepdim=True)], -1))  # 添加分组节点
            if x_base.size(0) == 0:  # 如果没有剩余节点
                x_base = data.x[:1] * 0  # 添加空节点
            data.x = x_base  # 更新基础特征
            data.xg = torch.cat(super_nodes, 0).view((1, -1))  # 合并超级节点特征

        return data  # 返回更新后的数据

    def compute_degree(self, edge_index, num_nodes):
        row, col = edge_index  # 提取边的起始和结束节点索引
        deg = degree(row, num_nodes)  # 计算每个节点的度
        deg = deg.view((-1, 1))  # 将度展平为列向量

        if self.onehot_maxdeg is not None and self.onehot_maxdeg > 0:  # 如果设置了one-hot度特征
            max_deg = torch.tensor(self.onehot_maxdeg, dtype=deg.dtype)  # 获取最大度
            deg_capped = torch.min(deg, max_deg).type(torch.int64)  # 将度限制在最大度
            deg_onehot = F.one_hot(deg_capped.view(-1), num_classes=self.onehot_maxdeg + 1)  # 计算one-hot编码
            deg_onehot = deg_onehot.type(deg.dtype)  # 转换one-hot编码的数据类型
        else:
            deg_onehot = self.empty_feature(num_nodes)  # 如果不使用one-hot度特征，则返回空特征

        if not self.degree:  # 如果不使用度特征
            deg = self.empty_feature(num_nodes)  # 返回空特征

        return deg, deg_onehot  # 返回度特征和one-hot度特征

    def compute_centrality(self, data):
        if not self.centrality:  # 如果不使用中心性特征
            return self.empty_feature(data.num_nodes)  # 返回空特征

        G = nx.Graph(data.edge_index.numpy().T.tolist())  # 使用边索引创建NetworkX图
        G.add_nodes_from(range(data.num_nodes))  # 确保所有节点都在图中
        closeness = nx.algorithms.closeness_centrality(G)  # 计算节点的接近中心性
        betweenness = nx.algorithms.betweenness_centrality(G)  # 计算节点的中介中心性
        pagerank = nx.pagerank_numpy(G)  # 计算节点的PageRank
        centrality_features = torch.tensor(
            [[closeness[i], betweenness[i], pagerank[i]] for i in range(data.num_nodes)]
        )  # 将中心性特征转换为张量
        return centrality_features  # 返回中心性特征

    def compute_akx(self, num_nodes, x, edge_index, edge_weight=None):
        if self.AK is None or self.AK <= 0:  # 如果不使用a^kx特征
            return self.empty_feature(num_nodes)  # 返回空特征

        edge_index, norm = self.norm(edge_index, num_nodes, edge_weight, diag_val=self.edge_norm_diag)  # 计算归一化边索引和归一化系数

        xs = []  # 初始化a^kx特征列表
        for k in range(1, self.AK + 1):  # 计算a^kx特征
            x = self.propagate(edge_index, x=x, norm=norm)  # 通过消息传递计算特征
            xs.append(x)  # 添加特征到列表
        return torch.cat(xs, -1)  # 合并所有a^kx特征

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j  # 计算消息传递的特征值

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, diag_val=1e-8, dtype=None):
        if edge_weight is None:  # 如果没有边权重
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)  # 初始化边权重为1
        edge_weight = edge_weight.view(-1)  # 展平边权重
        assert edge_weight.size(0) == edge_index.size(1)  # 确保边权重和边索引数量匹配

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)  # 移除自环边
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)  # 添加自环边
        # 为自环边添加边权重
        loop_weight = torch.full((num_nodes,), diag_val, dtype=edge_weight.dtype, device=edge_weight.device)  # 初始化自环边权重
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)  # 合并边权重和自环边权重

        row, col = edge_index  # 提取边的起始和结束节点索引
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)  # 计算每个节点的度
        deg_inv_sqrt = deg.pow(-0.5)  # 计算度的负0.5次方
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # 将无穷大值替换为0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]  # 返回归一化边索引和归一化系数

    def empty_feature(self, num_nodes):
        return torch.zeros([num_nodes, 0])  # 返回空特征张量
