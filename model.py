import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv

# MHGNIMC模型
class MHGNIMC(nn.Module):
    def __init__(self, m_emd, d_emd, nimc, args):
        super(MHGNIMC, self).__init__()
        self.args = args
        self.Xm = m_emd
        self.Xd = d_emd
        self.nimc = nimc

    def forward(self, dataset):
        Em = self.Xm(dataset) # (853, 128)
        Ed = self.Xd(dataset) # (591, 128)
        # pre_asso = Em.mm(Ed.t())
        pre_asso = self.nimc(Em, Ed)
        return pre_asso

# 神经归纳矩阵补全
class NIMC(nn.Module):
    def __init__(self, args):
        super(NIMC, self).__init__()
        self.args = args
        self.linear_x_1 = nn.Linear(128, 64) # (64, 32)
        self.linear_x_2 = nn.Linear(64, 32)

        self.linear_y_1 = nn.Linear(128, 64)
        self.linear_y_2 = nn.Linear(64, 32)


    def forward(self, X, Y):
        x1 = torch.relu(self.linear_x_1(X))
        x = torch.relu(self.linear_x_2(x1))
        y1 = torch.relu(self.linear_y_1(Y))
        y = torch.relu(self.linear_y_2(y1))
        return x.mm(y.t())


# 提取miRNA特征
class EmbeddingM(nn.Module):
    def __init__(self, args):
        super(EmbeddingM, self).__init__()
        self.args = args

        self.gcn_x1_f = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x2_f = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x1_s = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x2_s = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x1_g = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x2_g = GCNConv(self.args.fm, self.args.fm)

        self.gat_x_f = GATConv(self.args.fm, self.args.fm, heads=2, concat=False, edge_dim=1)
        self.gat_x_s = GATConv(self.args.fm, self.args.fm, heads=2, concat=False, edge_dim=1)
        self.gat_x_g = GATConv(self.args.fm, self.args.fm, heads=2, concat=False, edge_dim=1)

        self.fc1_x = nn.Linear(in_features=self.args.view * self.args.gcn_layers, out_features=5 * self.args.view * self.args.gcn_layers)
        self.fc2_x = nn.Linear(in_features=5 * self.args.view * self.args.gcn_layers, out_features=self.args.view * self.args.gcn_layers)
        self.sigmoidx = nn.Sigmoid()
        self.cnn_x = nn.Conv2d(in_channels=self.args.view * self.args.gcn_layers, out_channels=1, kernel_size=(1, 1), stride=1, bias=True)

    def forward(self, data):
        torch.manual_seed(1)
        x_m = torch.randn(self.args.miRNA_number, self.args.fm) # 随机初始化特征矩阵 (853, 128)

        x_m_f1 = torch.relu(self.gcn_x1_f(x_m.cuda(), data['mm_f']['edges'].cuda(), data['mm_f']['data_matrix'][
            data['mm_f']['edges'][0], data['mm_f']['edges'][1]].cuda()))
        x_m_f2 = torch.relu(self.gat_x_f(x_m_f1, data['mm_f']['edges'].cuda(), data['mm_f']['data_matrix'][
            data['mm_f']['edges'][0], data['mm_f']['edges'][1]].cuda()))
        x_m_f3 = torch.relu(self.gcn_x2_f(x_m_f1, data['mm_f']['edges'].cuda(), data['mm_f']['data_matrix'][
            data['mm_f']['edges'][0], data['mm_f']['edges'][1]].cuda()))

        x_m_s1 = torch.relu(self.gcn_x1_s(x_m.cuda(), data['mm_s']['edges'].cuda(), data['mm_s']['data_matrix'][
            data['mm_s']['edges'][0], data['mm_s']['edges'][1]].cuda()))
        x_m_s2 = torch.relu(self.gat_x_s(x_m_s1, data['mm_s']['edges'].cuda(), data['mm_s']['data_matrix'][
            data['mm_s']['edges'][0], data['mm_s']['edges'][1]].cuda()))
        x_m_s3 = torch.relu(self.gcn_x2_s(x_m_s1, data['mm_s']['edges'].cuda(), data['mm_s']['data_matrix'][
            data['mm_s']['edges'][0], data['mm_s']['edges'][1]].cuda()))

        x_m_g1 = torch.relu(self.gcn_x1_g(x_m.cuda(), data['mm_g']['edges'].cuda(), data['mm_g']['data_matrix'][
            data['mm_g']['edges'][0], data['mm_g']['edges'][1]].cuda()))
        x_m_g2 = torch.relu(self.gat_x_g(x_m_g1, data['mm_g']['edges'].cuda(), data['mm_g']['data_matrix'][
            data['mm_g']['edges'][0], data['mm_g']['edges'][1]].cuda()))
        x_m_g3 = torch.relu(self.gcn_x2_g(x_m_g1, data['mm_g']['edges'].cuda(), data['mm_g']['data_matrix'][
            data['mm_g']['edges'][0], data['mm_g']['edges'][1]].cuda()))

        # XM = torch.cat((x_m_f3, x_m_s3, x_m_g3), 1).t()
        # XM = torch.cat((x_m_f3, x_m_f1, x_m_s3, x_m_s1, x_m_g3, x_m_g1), 1).t()
        XM = torch.cat((x_m_f3, x_m_f2, x_m_f1, x_m_s3, x_m_s2, x_m_s1, x_m_g3, x_m_g2, x_m_g1), 1).t() # (768, 853)
        # XM = torch.cat((x_m_g3, x_m_g2, x_m_g1), 1).t()
        # XM = torch.cat((x_m_f4, x_m_f3, x_m_f2, x_m_f1, x_m_s4, x_m_s3, x_m_s2, x_m_s1, x_m_g4, x_m_g3, x_m_g2, x_m_g1), 1).t()
        XM = XM.view(1, self.args.view * self.args.gcn_layers, self.args.fm, -1) # (1, 6, 128, 853)

        globalAvgPool_x = nn.AvgPool2d((self.args.fm, self.args.miRNA_number), (1, 1))
        x_channel_attention = globalAvgPool_x(XM) # (1, 6, 1, 1)

        x_channel_attention = x_channel_attention.view(x_channel_attention.size(0), -1) # (1, 6)
        x_channel_attention = self.fc1_x(x_channel_attention) # (1, 30)
        x_channel_attention = torch.relu(x_channel_attention) # (1, 30)
        x_channel_attention = self.fc2_x(x_channel_attention) # (1, 6)
        x_channel_attention = self.sigmoidx(x_channel_attention) # (1, 6)
        x_channel_attention = x_channel_attention.view(x_channel_attention.size(0), x_channel_attention.size(1), 1, 1) # (1, 6, 1, 1)
        XM_channel_attention = x_channel_attention * XM # (1, 6, 128, 853)
        XM_channel_attention = torch.relu(XM_channel_attention) # (1, 6, 128, 853)

        x = self.cnn_x(XM_channel_attention) # (1, 1, 128, 853)
        x = x.view(self.args.fm, self.args.miRNA_number).t() # (853, 128)

        return x

# 提取疾病特征
class EmbeddingD(nn.Module):
    def __init__(self, args):
        super(EmbeddingD, self).__init__()
        self.args = args

        self.gcn_y1_f = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y2_f = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y1_s = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y2_s = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y1_g = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y2_g = GCNConv(self.args.fd, self.args.fd)

        self.gat_y_f = GATConv(self.args.fd, self.args.fd, heads=2, concat=False, edge_dim=1)
        self.gat_y_s = GATConv(self.args.fd, self.args.fd, heads=2, concat=False, edge_dim=1)
        self.gat_y_g = GATConv(self.args.fd, self.args.fd, heads=2, concat=False, edge_dim=1)

        self.fc1_y = nn.Linear(in_features=self.args.view * self.args.gcn_layers, out_features=5 * self.args.view  * self.args.gcn_layers)
        self.fc2_y = nn.Linear(in_features=5 * self.args.view  * self.args.gcn_layers, out_features=self.args.view  * self.args.gcn_layers)
        self.sigmoidy = nn.Sigmoid()
        self.cnn_y = nn.Conv2d(in_channels=self.args.view * self.args.gcn_layers, out_channels=1, kernel_size=(1, 1), stride=1, bias=True)

    def forward(self, data):
        torch.manual_seed(1)
        y_d = torch.randn(self.args.disease_number, self.args.fd)

        y_d_s1 = torch.relu(self.gcn_y1_s(y_d.cuda(), data['dd_s']['edges'].cuda(), data['dd_s']['data_matrix'][
            data['dd_s']['edges'][0], data['dd_s']['edges'][1]].cuda()))
        y_d_s2 = torch.relu(self.gat_y_s(y_d_s1, data['dd_s']['edges'].cuda(), data['dd_s']['data_matrix'][
            data['dd_s']['edges'][0], data['dd_s']['edges'][1]].cuda()))
        y_d_s3 = torch.relu(self.gcn_y2_s(y_d_s1, data['dd_s']['edges'].cuda(), data['dd_s']['data_matrix'][
            data['dd_s']['edges'][0], data['dd_s']['edges'][1]].cuda()))

        y_d_f1 = torch.relu(self.gcn_y1_f(y_d.cuda(), data['dd_f']['edges'].cuda(), data['dd_f']['data_matrix'][
            data['dd_f']['edges'][0], data['dd_f']['edges'][1]].cuda()))
        y_d_f2 = torch.relu(self.gat_y_f(y_d_f1, data['dd_f']['edges'].cuda(), data['dd_f']['data_matrix'][
            data['dd_f']['edges'][0], data['dd_f']['edges'][1]].cuda()))
        y_d_f3 = torch.relu(self.gcn_y2_f(y_d_f1, data['dd_f']['edges'].cuda(), data['dd_f']['data_matrix'][
            data['dd_f']['edges'][0], data['dd_f']['edges'][1]].cuda()))

        y_d_g1 = torch.relu(self.gcn_y1_g(y_d.cuda(), data['dd_g']['edges'].cuda(), data['dd_g']['data_matrix'][
            data['dd_g']['edges'][0], data['dd_g']['edges'][1]].cuda()))
        y_d_g2 = torch.relu(self.gat_y_g(y_d_g1, data['dd_g']['edges'].cuda(), data['dd_g']['data_matrix'][
            data['dd_g']['edges'][0], data['dd_g']['edges'][1]].cuda()))
        y_d_g3 = torch.relu(self.gcn_y2_g(y_d_g1, data['dd_g']['edges'].cuda(), data['dd_g']['data_matrix'][
            data['dd_g']['edges'][0], data['dd_g']['edges'][1]].cuda()))

        # YD = torch.cat((y_d_s3, y_d_f3, y_d_g3), 1).t()
        # YD = torch.cat((y_d_s3, y_d_s1, y_d_f3, y_d_f1, y_d_g3, y_d_g1), 1).t()
        YD = torch.cat((y_d_s3, y_d_s2, y_d_s1, y_d_f3, y_d_f2, y_d_f1, y_d_g3, y_d_g2, y_d_g1), 1).t()
        # YD = torch.cat((y_d_g3, y_d_g2, y_d_g1), 1).t()
        # YD = torch.cat((y_d_s4, y_d_s3, y_d_s2, y_d_s1, y_d_f4, y_d_f3, y_d_f2, y_d_f1, y_d_g4, y_d_g3, y_d_g2, y_d_g1), 1).t()
        YD = YD.view(1, self.args.view  * self.args.gcn_layers, self.args.fd, -1)

        globalAvgPool_y = nn.AvgPool2d((self.args.fm, self.args.disease_number), (1, 1))
        y_channel_attention = globalAvgPool_y(YD)

        y_channel_attention = y_channel_attention.view(y_channel_attention.size(0), -1)
        y_channel_attention = self.fc1_y(y_channel_attention)
        y_channel_attention = torch.relu(y_channel_attention)
        y_channel_attention = self.fc2_y(y_channel_attention)
        y_channel_attention = self.sigmoidy(y_channel_attention)
        y_channel_attention = y_channel_attention.view(y_channel_attention.size(0), y_channel_attention.size(1), 1, 1)

        YD_channel_attention = y_channel_attention * YD
        YD_channel_attention = torch.relu(YD_channel_attention)

        y = self.cnn_y(YD_channel_attention)
        y = y.view(self.args.fd, self.args.disease_number).t()

        return y
