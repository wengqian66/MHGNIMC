import torch
import random
from util import *
from datapro import *
from model import MHGNIMC, EmbeddingM, EmbeddingD, NIMC
import numpy as np

# 五折交叉验证
def train_test(dataset, args, random_seed):
    dataset['true'] = read_csv('datasets/multi_data/m_d.csv')
    args.miRNA_number = dataset['true'].shape[0]
    args.disease_number = dataset['true'].shape[1]
    five_test_metric = []
    test_metric = []
    kfolds = args.kfold
    for foldind in range(kfolds):
        print(f'################Fold {foldind + 1} of {kfolds}################')
        # 处理正负样本
        one_index = []
        one_train_index = []
        one_test_index = []
        zero_index = []
        for i in range(args.miRNA_number):
            for j in range(args.disease_number):
                if dataset['true'][i][j] >= 1:
                    one_index.append([i, j])
                else:
                    zero_index.append([i, j])
        # random.seed(args.random_seed)
        random.seed(random_seed)
        random.shuffle(one_index)
        random.shuffle(zero_index)
        one_test_index = one_index[foldind * int(len(one_index) * 0.2):(foldind + 1) * int(len(one_index) * 0.2)]
        [one_train_index.append(i) for i in one_index if i not in one_test_index]
        if args.negative_rate == -1:
            zero_index = zero_index
        else:
            zero_train_index = zero_index[:int(args.negative_rate * len(one_train_index))]
            zero_test_index = zero_index[int(len(one_train_index)):(
                        int(len(one_train_index)) + int(args.negative_rate * len(one_test_index)))]

        Y_train = np.zeros([args.miRNA_number, args.disease_number], dtype=int)
        Y_test = np.zeros([args.miRNA_number, args.disease_number], dtype=int)
        one_train_index_T = list(map(list, zip(*one_train_index)))
        one_test_index_T = list(map(list, zip(*one_test_index)))
        for i in range(len(one_train_index)):
            Y_train[one_train_index_T[0][i], one_train_index_T[1][i]] = 1
        for j in range(len(one_test_index)):
            Y_test[one_test_index_T[0][j], one_test_index_T[1][j]] = 1
        dataset['Y_train'] = Y_train
        dataset['Y_test'] = Y_test

        # 生成高斯相似性矩阵
        "miRNA Gaussian sim"
        mm_g_matrix = calculate_gauss_sim(Y_train, axis=0)
        mm_g_matrix = torch.tensor(mm_g_matrix, dtype=torch.float32)
        mm_g_edge_index = get_edge_index(mm_g_matrix)
        dataset['mm_g'] = {'data_matrix': mm_g_matrix, 'edges': mm_g_edge_index}

        "disease Gaussian sim"
        dd_g_matrix = calculate_gauss_sim(Y_train, axis=1)
        dd_g_matrix = torch.tensor(dd_g_matrix, dtype=torch.float32)
        dd_g_edge_index = get_edge_index(dd_g_matrix)
        dataset['dd_g'] = {'data_matrix': dd_g_matrix, 'edges': dd_g_edge_index}

        # 构建训练集
        train_edges = np.array(one_train_index + zero_train_index, dtype=int)  # (19914, 2)
        train_true_label = np.array([1] * len(one_train_index) + [0] * len(zero_train_index), dtype=np.float32)  # 19914

        # 构建测试集
        test_edges = np.array(one_test_index + zero_test_index, dtype=int)  # (4978, 2)
        test_edges_T = test_edges.T
        test_true_label = np.array([1] * len(one_test_index) + [0] * len(zero_test_index), dtype=np.float32) # 4978

        dataset['test_edge'] = torch.tensor(test_edges_T)
        dataset['test_label'] = torch.tensor(test_true_label)

        # 训练模型
        model = MHGNIMC(EmbeddingM(args), EmbeddingD(args), NIMC(args), args)
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0) # 0.001 , 0.0
        print("-----training-----")
        model.train()
        train_loss = Myloss(args)
        one_index = torch.tensor(one_train_index).cuda().t().tolist()
        zero_index = torch.tensor(zero_train_index).cuda().t().tolist()
        for e in range(args.epoch):
            pre_score = model(dataset)
            loss = train_loss(one_index, zero_index, dataset['true'].cuda(), pre_score)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (e + 1) % 50 == 0 or e == 0:
                loss_str = loss.item() / (len(one_index[0]) + len(zero_index[0]))
                print('------EOPCH {} of {}------'.format(e + 1, args.epoch))
                print('loss:{}'.format(loss_str))

        # 测试模型
        model.eval()
        with torch.no_grad():
            print("-----testing-----")
            pre_score = model(dataset)
            pred = pre_score[dataset['test_edge'][0], dataset['test_edge'][1]].cpu().detach().numpy()
            target = dataset['test_label'].numpy()
            metric = get_metrics(pred, target)
            test_metric.append(metric)

    cv_metric = np.mean(test_metric, axis=0)
    cv_metric_list = cv_metric.tolist()
    print_met(cv_metric)
    return cv_metric_list