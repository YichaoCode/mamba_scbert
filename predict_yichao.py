# -*- coding: utf-8 -*-
import os
import gc
import argparse
import json
import random
import math
import random
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.nn import functional as F
from performer_pytorch import PerformerLM
import scanpy as sc
import anndata as ad
from utils import *
import pickle as pkl
import logging

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
parser.add_argument("--novel_type", type=bool, default=False, help='Novel cell type exists or not.')
parser.add_argument("--unassign_thres", type=float, default=0.5, help='The confidence score threshold for novel cell type annotation.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
parser.add_argument("--data_path", type=str, default='./data/Zheng68K.h5ad', help='Path of data for predicting.')
parser.add_argument("--model_path", type=str, default='./ckpts/finetune_best.pth', help='Path of finetuned model.')

args = parser.parse_args()
logging.info(f"Command line arguments: {args}")

SEED = args.seed
EPOCHS = args.epoch
SEQ_LEN = args.gene_num + 1
UNASSIGN = args.novel_type
UNASSIGN_THRES = args.unassign_thres if UNASSIGN == True else 0

CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

class Identity(torch.nn.Module):
    def __init__(self, dropout = 0., h_dim = 100, out_dim = 10):
        super(Identity, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=SEQ_LEN, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

logging.info("Loading data...")
data = sc.read_h5ad(args.data_path)
logging.info(f"Data loaded. Shape: {data.shape}")

with open('label_dict', 'rb') as fp:
    label_dict = pkl.load(fp)
with open('label', 'rb') as fp:
    label = pkl.load(fp)
logging.info(f"Label dictionary and labels loaded. Label dictionary size: {len(label_dict)}, Labels shape: {label.shape}")

class_num = np.unique(label, return_counts=True)[1].tolist()
class_weight = torch.tensor([(1 - (x / sum(class_num))) ** 2 for x in class_num])
label = torch.from_numpy(label)
data = data.X
logging.info(f"Data processed. Data shape: {data.shape}")

logging.info("Loading model...")
model = PerformerLM(
    num_tokens = CLASS,
    dim = 200,
    depth = 6,
    max_seq_len = SEQ_LEN,
    heads = 10,
    local_attn_heads = 0,
    g2v_position_emb = True
)
model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])

path = args.model_path
ckpt = torch.load(path)
model.load_state_dict(ckpt['model_state_dict'])
for param in model.parameters():
    param.requires_grad = False
model = model.to(device)
logging.info("Model loaded and set to evaluation mode.")

def predict(data, model, device, label_dict, UNASSIGN_THRES, CLASS, batch_size):
    logging.info(f"Predicting for data with shape: {data.shape}")
    model.eval()
    pred_finals = []
    novel_indices = []
    with torch.no_grad():
        for index in range(batch_size):
            logging.info(f"Processing cell {index+1}/{batch_size}")
            
            # 获取当前细胞的基因表达向量
            full_seq = data[index]  # 直接使用 data[index],不调用 toarray() 方法
            logging.info(f"Original gene expression vector shape: {full_seq.shape}")
            
            # 将表达值大于 CLASS-2 的基因表达值设置为 CLASS-2
            full_seq[full_seq > (CLASS - 2)] = CLASS - 2
            logging.info(f"Gene expression vector shape after thresholding: {full_seq.shape}")
            
            # 将基因表达向量转换为 PyTorch 张量
            full_seq = torch.from_numpy(full_seq).long()
            logging.info(f"Gene expression vector shape after converting to tensor: {full_seq.shape}")
            logging.info(f"Gene expression vector data type: {full_seq.dtype}")
            
            # 检查基因表达向量中的最大值是否超出词嵌入层大小
            max_token = torch.max(full_seq).item()
            logging.info(f"Maximum token value in the gene expression vector: {max_token}")
            
            # 在基因表达向量的末尾添加一个零,表示序列的结束
            full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
            logging.info(f"Gene expression vector shape after appending zero: {full_seq.shape}")
            
            # 为基因表达向量添加一个维度,表示批次大小为1
            full_seq = full_seq.unsqueeze(0)
            logging.info(f"Gene expression vector shape after unsqueezing: {full_seq.shape}")
            
            # 使用模型对基因表达向量进行预测
            logging.info(f"Passing gene expression vector to the model...")
            pred_logits = model(full_seq)
            logging.info(f"Prediction logits shape: {pred_logits.shape}")
            
            # 对预测结果应用 softmax 函数,得到各个类别的概率
            softmax = nn.Softmax(dim=-1)
            pred_prob = softmax(pred_logits)
            logging.info(f"Prediction probabilities shape: {pred_prob.shape}")
            
            # 选择概率最大的类别作为最终预测结果
            pred_final = pred_prob.argmax(dim=-1).item()
            logging.info(f"Predicted class index: {pred_final}")
            
            # 如果最大概率低于阈值,则将该细胞标记为 "Unassigned"
            if np.amax(np.array(pred_prob.cpu()), axis=-1) < UNASSIGN_THRES:
                novel_indices.append(index)
                logging.info(f"Cell {index+1} is marked as 'Unassigned' due to low prediction confidence.")
            
            # 将预测结果添加到列表中
            pred_finals.append(pred_final)
            
    # 使用 label_dict 将预测的类别索引转换为对应的细胞类型标签        
    pred_list = label_dict[pred_finals].tolist()
    logging.info(f"Predicted cell types: {pred_list}")
    
    # 将 "Unassigned" 的细胞类型标签添加到预测结果中
    for index in novel_indices:
        pred_list[index] = 'Unassigned'
    logging.info(f"Final predicted cell types (including 'Unassigned'): {pred_list}")
    
    logging.info("Prediction completed.")
    return pred_list

logging.info("Predicting...")
batch_size = data.shape[0]
pred_list = predict(data, model, device, label_dict, UNASSIGN_THRES, CLASS, batch_size)

logging.info(f"Final predicted cell types: {pred_list}")
