import random
import numpy as np
from datapro import Simdata_pro
from train import train_test
from util import *

class Config:
    def __init__(self):
        self.alpha = 0.4 # 0.4 # 损失函数中正负样本的权重
        self.kfold = 5 # 五折交叉验证
        self.epoch = 260 # 260 # 训练轮数
        self.gcn_layers = 3 # GNN层数
        self.view = 3 # 视角数
        self.fm = 128 # 128 # miRNA隐藏层层数
        self.fd = 128 # 128 # 疾病隐藏层层数
        self.random_seed = [74, 102, 85, 145, 176] # 随机种子
        self.negative_rate = 1 # 正负样本比例

def main():
    args = Config()
    # 打印参数
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    dataset = Simdata_pro(args)
    five_test_metric_list = []
    # 五次五折验证
    for random_seed in args.random_seed:
        print("------------第{}次五折验证------------".format(args.random_seed.index(random_seed) + 1))
        five_test_metric = train_test(dataset, args, random_seed)
        five_test_metric_list.append(five_test_metric)
    five_test_metric_list = np.array(five_test_metric_list)
    cv_five_test_metric = np.mean(five_test_metric_list, axis=0)
    print(np.std(five_test_metric_list, axis=0))
    print("------------{}次五折的平均结果------------".format(len(args.random_seed)))
    print_met(cv_five_test_metric)


if __name__ == "__main__":
    main()
