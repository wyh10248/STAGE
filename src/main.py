import numpy as np
import torch
import pandas as pd
from TRDEMDA import *
from sklearn.model_selection import KFold
from clac_metric import get_metrics
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import random
from utils import *
import gc
import matplotlib.pyplot as plt
import argparse
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
dataset = 'HMDAD' 
#dataset = 'peryton'   
if dataset == 'HMDAD':
    embed_dim_default = 331
    max_len_x = 500
    epoch =210
    seed = 50
    a = 0.8
    b = 0.1
elif dataset == 'peryton':
    embed_dim_default = 1439
    max_len_x = 1500
    epoch = 170
    seed = 42
    a = 0.5
    b = 0.4
else:
    raise ValueError(f"Unknown dataset: {dataset}")
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.001, type=float,help='learning rate')
parser.add_argument('--seed', type=int, default=seed, help='Random seed.')
parser.add_argument('--k_fold', type=int, default=5, help='crossval_number.')
parser.add_argument('--epoch', type=int, default=epoch, help='train_number.')
parser.add_argument('--in_dim', type=int, default=embed_dim_default, help='in_feature.')
parser.add_argument('--hidden_dim', type=int, default=256, help='hidden_dim.')
parser.add_argument('--latent_dim', type=int, default=128, help='latent_dim.')
parser.add_argument('--num_layers', type=int, default=3, help='num_layers.')
parser.add_argument('--embed_dim', type=int, default=embed_dim_default, help='embed_dim.')
parser.add_argument('--num_heads', type=int, default=8, help='head_number.')
parser.add_argument('--output_dim', type=int, default=256, help='output_dim.')
parser.add_argument('--drop_rate', type=int, default=0.2, help='drop_rate.')
parser.add_argument('--max_len', type=int, default=max_len_x, help='max_len.')
args = parser.parse_args()

device = torch.device("cpu")

def crossval_index(drug_mic_matrix, sizes):
    random.seed(sizes.seed)
    set_seed(args.seed)
    pos_index_matrix = np.mat(np.where(drug_mic_matrix == 1))
    neg_index_matrix = np.mat(np.where(drug_mic_matrix == 0))
    pos_index = random_index(neg_index_matrix, sizes)
    neg_index = random_index(pos_index_matrix, sizes)
    index = []
    for i in range(sizes.k_fold):
        # 对每一折的正负样本进行平衡处理
        balanced_pos_index, balanced_neg_index = balance_samples(pos_index[i], neg_index[i])
        index.append(balanced_pos_index + balanced_neg_index)
    return index

def cross_validation_experiment(A, microSimi, disSimi, args):
    index = crossval_index(A, args)
    metric = np.zeros((1, 7))
    score =[]
    tprs=[]
    fprs=[]
    aucs=[]
    precisions=[]
    recalls = []
    auprs = []
    pre_matrix = np.zeros(A.shape)
    print("seed=%d, evaluating lncRNA-disease...." % (args.seed))
    for k in range(args.k_fold):
        print("------this is %dth cross validation------" % (k + 1))
        
        train_matrix = np.matrix(A, copy=True)
        train_matrix[tuple(np.array(index[k]).T)] = 0  # 将5折中的一折变为0
        x=constructHNet(train_matrix, microSimi, disSimi)
        edge_index=adjacency_matrix_to_edge_index(train_matrix)
        
    
        microbe_len = A.shape[0]
        dis_len = A.shape[1]
        
        train_matrix = torch.from_numpy(train_matrix).to(torch.float32)
        x = x.to(torch.float32)
        edge_index = edge_index.to(torch.int64)
        
        model = TASEMDA(args).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

        #训练循环
        model.train()
        for epoch in range(args.epoch):
            optimizer.zero_grad()
            pred, label = model(args, x, edge_index, train_matrix, train_model=True)
            loss = F.binary_cross_entropy(pred.float(), label)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

         # 模型评估
        model.eval()
        with torch.no_grad():
            pred_scores, _ = model(args, x, edge_index, train_matrix, train_model=False)

        micro_dis_res= pred_scores.detach().cpu()
        predict_y_proba = micro_dis_res.reshape(microbe_len, dis_len).detach().numpy()
        pre_matrix[tuple(np.array(index[k]).T)] = predict_y_proba[tuple(np.array(index[k]).T)]  #从预测分数矩阵中取出验证集的预测结果 只返回相应的预测分数
        A = np.array(A)
        metric_tmp = get_metrics(A[tuple(np.array(index[k]).T)],
                                  predict_y_proba[tuple(np.array(index[k]).T)]) #预测结果所得的评价指标
        fpr, tpr, t = roc_curve(A[tuple(np.array(index[k]).T)],
                                  predict_y_proba[tuple(np.array(index[k]).T)])
        precision, recall, _ = precision_recall_curve(A[tuple(np.array(index[k]).T)],
                                  predict_y_proba[tuple(np.array(index[k]).T)])
        tprs.append(tpr)
        fprs.append(fpr)
        precisions.append(precision)
        recalls.append(recall)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        auprs.append(metric_tmp[1])
        print(metric_tmp)
        metric += metric_tmp      #  五折交叉验证的结果求和
        score.append(pre_matrix)
        del train_matrix  # del只删除变量，不删除数据
        gc.collect()  # 垃圾回收
    print('Mean:', metric / args.k_fold)
    metric = np.array(metric / args.k_fold)   #  五折交叉验证的结果求均值
    return metric, score, microbe_len, dis_len, tprs, fprs, aucs, precisions, recalls, auprs
        

def main(args):
# #------data1---------
    data_path = '../dataset/'
    data_set = 'HMDAD/'#292*39

    A = np.loadtxt(data_path + data_set + 'A.csv',delimiter=',')
    disSi = np.loadtxt(data_path + data_set + 'disfunsim.csv',delimiter=',') 
    disG= np.loadtxt(data_path + data_set + 'GSD.csv',delimiter=',') 
    micfu = np.loadtxt(data_path + data_set + 'microfunsim.csv',delimiter=',')
    micG = np.loadtxt(data_path + data_set + 'GSM.csv',delimiter=',')
    MIS = a*micfu + (1-a)*micG
    DIS = b*disSi + (1-b)*disG
#------data2---------
    # data_path = '../dataset/'
    # data_set = 'peryton/'#1396*43

    # A = np.loadtxt(data_path + data_set + 'A.csv', delimiter=',')
    # disSi = np.loadtxt(data_path + data_set + 'disfunsim.csv',delimiter=',') 
    # disG= np.loadtxt(data_path + data_set + 'GSD.csv',delimiter=',') 
    # micfu = np.loadtxt(data_path + data_set + 'microfunsim.csv',delimiter=',')
    # micG = np.loadtxt(data_path + data_set + 'GSM.csv',delimiter=',')
    # MIS = a*micfu + (1-a)*micG
    # DIS = b*disSi + (1-b)*disG
    
#-------------------------
    result, score, microbe_len, dis_len, tprs, fprs, aucs, precisions, recalls, auprs = cross_validation_experiment(A, MIS, DIS, args)
    final_score = np.zeros_like(A, dtype=np.float32)
    for fold_score in score:
        final_score += fold_score
    # 转换为 DataFrame（行列与原始矩阵一致）
    # df = pd.DataFrame(final_score)
    
    # # 保存为 CSV 文件（不保存索引和列名）
    # output_filename = f"{dataset}_prediction_score.csv"
    # df.to_csv(output_filename, index=False, header=False)
    sizes = Sizes(microbe_len, dis_len)
    score_matrix = np.mean(score, axis=0)
    print(list(sizes.__dict__.values()) + result.tolist()[0][:2])
    plot_auc_curves(fprs, tprs, aucs)
    plot_prc_curves(precisions, recalls, auprs)
    


if __name__== '__main__':
    main(args)


 