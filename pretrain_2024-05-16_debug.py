# -*- coding: utf-8 -*-

import logging
import os
import gc
import argparse
import json
import random
import math
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from performer_pytorch import PerformerLM
import scanpy as sc
import anndata as ad
from utils import *
import warnings
from datetime import datetime


warnings.filterwarnings("ignore", category=UserWarning)  # 忽略 UserWarning 类型的警告 临时忽略异常


# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, help='Local process rank.')
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
parser.add_argument("--batch_size", type=int, default=4, help='Number of batch size.')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
parser.add_argument("--mask_prob", type=float, default=0.15, help='Probability of masking.')
parser.add_argument("--replace_prob", type=float, default=0.9, help='Probability of replacing with [MASK] token for masking.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
parser.add_argument("--data_path", type=str, default='./data/panglao_human.h5ad', help='Path of data for pretraining.')
parser.add_argument("--ckpt_dir", type=str, default='./ckpts/', help='Directory of checkpoint to save.')
parser.add_argument("--model_name", type=str, default='panglao_pretrain', help='Pretrained model name.')

args = parser.parse_args()



SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
LEARNING_RATE = args.learning_rate
SEQ_LEN = args.gene_num + 1
VALIDATE_EVERY = args.valid_every
CLASS = args.bin_num + 2
MASK_PROB = args.mask_prob
REPLACE_PROB = args.replace_prob
RANDOM_TOKEN_PROB = 0.
MASK_TOKEN_ID = CLASS - 1
PAD_TOKEN_ID = CLASS - 1
MASK_IGNORE_TOKEN_IDS = [0]
POS_EMBED_USING = args.pos_embed

#数据遮蔽函数
def data_mask(data,
    mask_prob = MASK_PROB,
    replace_prob = REPLACE_PROB,
    num_tokens = None,
    random_token_prob = RANDOM_TOKEN_PROB,
    mask_token_id = MASK_TOKEN_ID,
    pad_token_id = PAD_TOKEN_ID,
    mask_ignore_token_ids = MASK_IGNORE_TOKEN_IDS
):
    mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])
    # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
    # also do not include these special tokens in the tokens chosen at random
    no_mask = mask_with_tokens(data, mask_ignore_token_ids)   # ignore_token as True, will not be masked later
    mask = get_mask_subset_with_prob(~no_mask, mask_prob)      # get the True/False mask matrix
    # get mask indices
    ## mask_indices = torch.nonzero(mask, as_tuple=True)   # get the index of mask(nonzero value of mask matrix)
    # mask input with mask tokens with probability of `replace_prob` (keep tokens the same with probability 1 - replace_prob)
    masked_input = data.clone().detach()
    # if random token probability > 0 for mlm
    if random_token_prob > 0:
        assert num_tokens is not None, 'num_tokens keyword must be supplied when instantiating MLM if using random token replacement'
        random_token_prob = prob_mask_like(data, random_token_prob)       # get the mask matrix of random token replace
        random_tokens = torch.randint(0, num_tokens, data.shape, device=data.device)     # generate random token matrix with the same shape as input
        random_no_mask = mask_with_tokens(random_tokens, mask_ignore_token_ids)        # not masked matrix for the random token matrix
        random_token_prob &= ~random_no_mask        # get the pure mask matrix of random token replace
        random_indices = torch.nonzero(random_token_prob, as_tuple=True)        # index of random token replace
        masked_input[random_indices] = random_tokens[random_indices]        # replace some tokens by random token
    # [mask] input
    replace_prob = prob_mask_like(data, replace_prob)     # get the mask matrix of token being masked
    masked_input = masked_input.masked_fill(mask * replace_prob, mask_token_id)        # get the data has been masked by mask_token
    # mask out any tokens to padding tokens that were not originally going to be masked
    labels = data.masked_fill(~mask, pad_token_id)        # the label of masked tokens
    return masked_input, labels

def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)      # num of mask of a single sequence in average
    num_tokens = mask.sum(dim=-1, keepdim=True)     # num of pure tokens of each sequence except special tokens
    mask_excess = torch.cat((torch.zeros(0), torch.arange(mask.size(-1)).repeat(mask.size(0)))).reshape(mask.size(0),mask.size(-1)).to(device)
    mask_excess = (mask_excess >= (num_tokens * prob).ceil())        # only 15% of pure tokens can be masked
    mask_excess = mask_excess[:, :max_masked]       # get difference between 15% of pure tokens and 15% of all tokens
    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)     # rand (0-1) as prob, special token use -1e9
    _, sampled_indices = rand.topk(max_masked, dim=-1)      # get index of topk prob to mask
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)        # delete difference of mask not pure
    new_mask = torch.zeros((batch, seq_len + 1), device=device)     # get (batch, seq_len) shape zero matrix
    new_mask.scatter_(-1, sampled_indices, 1)       # set masks in zero matrix as 1
    return new_mask[:, 1:].bool()       # the final mask, True is mask




# 初始化设置# 初始化设置
def initialize_settings(args):
    # 初始化进程组
    dist.init_process_group(backend='nccl')
    
    local_rank = args.local_rank if args.local_rank is not None else int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get("RANK", 0))
    is_master = rank == 0
    
    # 设置设备
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    # 获取世界大小和排名
    world_size = torch.distributed.get_world_size()
    world_rank = torch.distributed.get_rank()
    
    # 设置随机种子
    SEED = args.seed
    seed_value = SEED + world_rank
    seed_all(seed_value)
    
    # 配置日志
    hostname = os.uname()[1]
    logging.basicConfig(level=logging.INFO, format=f'%(asctime)s - [Process %(process)d] - [Host: {hostname}] - %(levelname)s  - [World Rank {world_rank}] - %(message)s')

    logging.info(f"Local rank: {local_rank}")
    logging.info(f"Device set to {device}")
    logging.info(f"World size: {world_size}")
    logging.info(f"World rank: {world_rank}")
    logging.info(f"Seed set to {seed_value} (Base Seed: {SEED}, World Rank: {world_rank})")
    
    # 返回需要的变量
    return device, world_size, is_master, args.model_name, args.ckpt_dir, SEED, args.batch_size, args.bin_num + 2, args.gene_num + 1, args.pos_embed, local_rank, args.epoch, args.learning_rate, args.bin_num + 2, args.grad_acc



class SCDataset(Dataset):
    used_indices = set()

    def __init__(self, data, CLASS, device):
        super().__init__()
        self.data = data
        self.CLASS = CLASS
        self.device = device

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0] - 1)
        while rand_start in SCDataset.used_indices:
            rand_start = random.randint(0, self.data.shape[0] - 1)
        SCDataset.used_indices.add(rand_start)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (self.CLASS - 2)] = self.CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(self.device)
        logging.info(f"[Process {dist.get_rank()}] - [GPU {self.device}] - Index {index} - Data index {rand_start}")
        return full_seq

    def __len__(self):
        return self.data.shape[0]



def prepare_data(args, SEED, BATCH_SIZE, device, world_size, CLASS):
    data = sc.read_h5ad(args.data_path)
    data = data.X
    data_train, data_val = train_test_split(data, test_size=0.05, random_state=SEED)

    train_dataset = SCDataset(data_train, CLASS, device)
    val_dataset = SCDataset(data_val, CLASS, device)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

    return train_loader, val_loader






# 确认分布式训练初始化
def print_dist_init_confirmation(local_rank):
    if dist.is_initialized():
        logging.info(f"Distributed training initialized. World size: {dist.get_world_size()}, "
                     f"Rank: {dist.get_rank()}, Local Rank: {local_rank}, "
                     f"Master: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
    else:
        logging.error("Failed to initialize distributed training.")


def record_distributed_operation(operation, *args, **kwargs):
    """
    包装分布式操作以记录执行时间和捕获异常。
    """
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    try:
        result = operation(*args, **kwargs)
        end_time.record()
        torch.cuda.synchronize()  # 等待记录的事件完成
        elapsed_time = start_time.elapsed_time(end_time)
        logging.info(f"{operation.__name__} completed in {elapsed_time:.2f}ms")
        return result
    except Exception as e:
        logging.error(f"Error during {operation.__name__}: {e}")
        raise


def compare_model_parameters(model):
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return

    local_params = torch.cat([p.data.view(-1) for p in model.parameters()])

    if dist.get_rank() == 0:
        global_params = [torch.zeros_like(local_params) for _ in range(dist.get_world_size())]
        record_distributed_operation(dist.gather, tensor=local_params, gather_list=global_params, dst=0)

        # 检查并记录每个节点参数的一致性
        is_consistent = True
        for i, params in enumerate(global_params):
            if not torch.equal(global_params[0], params):
                logging.warning(f"参数不一致: 节点0与节点{i}")
                is_consistent = False

        if is_consistent:
            logging.info("所有节点的模型参数一致。")
        else:
            logging.error("警告：检测到节点间模型参数不一致。")
    else:
        record_distributed_operation(dist.gather, tensor=local_params, dst=0)








# 模型构建
def build_model(args, device, CLASS, SEQ_LEN, POS_EMBED_USING, local_rank):
    model = PerformerLM(
        num_tokens=CLASS,
        dim=200,
        depth=6,
        max_seq_len=SEQ_LEN,
        heads=10,
        local_attn_heads=0,
        g2v_position_emb=POS_EMBED_USING
    )
    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    return model

# 训练循环# 训练循环import logging

# 训练循环
def train_loop(model, train_loader, val_loader, optimizer, scheduler, loss_fn, softmax, EPOCHS, VALIDATE_EVERY, GRADIENT_ACCUMULATION, device, world_size, is_master, local_rank, PAD_TOKEN_ID, model_name, ckpt_dir):
    dist.barrier()
    for epoch in range(1, EPOCHS + 1):
        train_loader.sampler.set_epoch(epoch)
        SCDataset.used_indices = set()
        model.train()
        dist.barrier()
        running_loss = 0.0
        cum_acc = 0.0
        for batch_idx, data in enumerate(train_loader, 1):
            logging.info(f"Process {local_rank}, Epoch {epoch}, Batch {batch_idx}: Data loaded to device {device}")
            data = data.to(device)
            data, labels = data_mask(data)
            logging.info(f"Process {local_rank}, Epoch {epoch}, Batch {batch_idx}: Data masked. Data: {data.cpu().numpy()}")
            if batch_idx % GRADIENT_ACCUMULATION != 0:
                with model.no_sync():
                    logits = model(data)
                    loss = loss_fn(logits.transpose(1, 2), labels) / GRADIENT_ACCUMULATION
                    loss.backward()
                    logging.info(f"Process {local_rank}, Epoch {epoch}, Batch {batch_idx}: Gradients calculated (no sync)")
            if batch_idx % GRADIENT_ACCUMULATION == 0:
                logits = model(data)
                loss = loss_fn(logits.transpose(1, 2), labels) / GRADIENT_ACCUMULATION
                loss.backward()
                logging.info(f"Process {local_rank}, Epoch {epoch}, Batch {batch_idx}: Gradients calculated")
                torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e2))
                logging.info(f"Process {local_rank}, Epoch {epoch}, Batch {batch_idx}: Gradients clipped")
                optimizer.step()


                compare_model_parameters(model)


                logging.info(f"Process {local_rank}, Epoch {epoch}, Batch {batch_idx}: Optimizer stepped")
                optimizer.zero_grad()
                logging.info(f"Process {local_rank}, Epoch {epoch}, Batch {batch_idx}: Gradients zeroed")
            running_loss += loss.item()
            final = softmax(logits)[..., 1:-1]
            final = final.argmax(dim=-1) + 1
            pred_num = (labels != PAD_TOKEN_ID).sum(dim=-1)
            correct_num = ((labels != PAD_TOKEN_ID) * (final == labels)).sum(dim=-1)
            cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
            logging.info(f"Process {local_rank}, Epoch {epoch}, Batch {batch_idx}: Loss: {loss.item():.6f}, Accuracy: {100 * torch.true_divide(correct_num, pred_num).mean().item():.4f}%")
        epoch_loss = running_loss / batch_idx
        epoch_acc = 100 * cum_acc / batch_idx
        epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
        epoch_acc = get_reduced(epoch_acc, local_rank, 0, world_size)
        if is_master:
            logging.info(f'Epoch {epoch} completed. Training Loss: {epoch_loss:.6f}, Accuracy: {epoch_acc:.4f}%')
        dist.barrier()
        scheduler.step()

        compare_model_parameters(model)

        logging.info(f"Process {local_rank}, Epoch {epoch}: Scheduler stepped")

        if epoch % VALIDATE_EVERY == 0:
            logging.info(f"Process {local_rank}, Epoch {epoch}: Starting validation")
            validate(model, val_loader, loss_fn, softmax, device, world_size, is_master, local_rank, PAD_TOKEN_ID, epoch)

        if is_master:
            logging.info(f"Process {local_rank}, Epoch {epoch}: Saving checkpoint")
            save_ckpt(epoch, model, optimizer, scheduler, epoch_loss, model_name, ckpt_dir, world_size)






# 验证
def validate(model, val_loader, loss_fn, softmax, device, world_size, is_master, local_rank, PAD_TOKEN_ID, epoch):
    model.eval()
    dist.barrier()
    running_loss = 0.0
    running_error = 0.0
    predictions = []
    truths = []
    with torch.no_grad():
        for index, data in enumerate(val_loader):
            index += 1
            data = data.to(device)
            data, labels = data_mask(data, PAD_TOKEN_ID=PAD_TOKEN_ID)
            logits = model(data)
            loss = loss_fn(logits.transpose(1, 2), labels)
            running_loss += loss.item()
            softmax = nn.Softmax(dim=-1)
            final = softmax(logits)[..., 1:-1]
            final = final.argmax(dim=-1) + 1
            predictions.append(final)
            truths.append(labels)
            logging.info(f"Process {local_rank}, Epoch {epoch}, Validation Batch {index}: Data: {data.cpu().numpy()}, Labels: {labels.cpu().numpy()}, Predictions: {final.cpu().numpy()}")
        del data, labels, logits, final
        # gather
        predictions = distributed_concat(torch.cat(predictions, dim=0), len(val_loader.dataset), world_size)
        truths = distributed_concat(torch.cat(truths, dim=0), len(val_loader.dataset), world_size)
        correct_num = ((truths != PAD_TOKEN_ID) * (predictions == truths)).sum(dim=-1)[0].item()
        val_num = (truths != PAD_TOKEN_ID).sum(dim=-1)[0].item()
        val_loss = running_loss / index
        val_loss = get_reduced(val_loss, local_rank, 0, world_size)
    if is_master:
        val_acc = 100 * correct_num / val_num
        logging.info(f'Process {local_rank}, Epoch {epoch}, Validation Loss: {val_loss:.6f}, Accuracy: {val_acc:6.4f}%')

# 主函数
def main():
    # 初始化设置
    # 调用 initialize_settings 函数，传入命令行参数 args
    # 返回值包括设备、世界大小、是否为主进程、模型名称、检查点目录、随机种子、批次大小、类别数、序列长度、是否使用位置嵌入、本地进程序号、训练轮数、学习率、填充标记 ID 和梯度累积步数
    device, world_size, is_master, model_name, ckpt_dir, SEED, BATCH_SIZE, CLASS, SEQ_LEN, POS_EMBED_USING, local_rank, EPOCHS, LEARNING_RATE, PAD_TOKEN_ID, GRADIENT_ACCUMULATION = initialize_settings(args)
    

    print_dist_init_confirmation(local_rank)





    # 准备数据
    # 调用 prepare_data 函数，传入命令行参数 args、随机种子、批次大小、设备、世界大小和类别数
    # 返回值包括训练数据加载器和验证数据加载器
    train_loader, val_loader = prepare_data(args, SEED, BATCH_SIZE, device, world_size, CLASS)

    # 构建模型
    # 调用 build_model 函数，传入命令行参数 args、设备、类别数、序列长度、是否使用位置嵌入和本地进程序号
    # 返回值为构建好的模型
    model = build_model(args, device, CLASS, SEQ_LEN, POS_EMBED_USING, local_rank)

    # 定义优化器、学习率调度器、损失函数和 softmax 函数
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=15,
        cycle_mult=2,
        max_lr=LEARNING_RATE,
        min_lr=1e-6,
        warmup_steps=5,
        gamma=0.9
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, reduction='mean').to(local_rank)
    softmax = nn.Softmax(dim=-1)

    # 训练循环
    # 调用 train_loop 函数，传入所有必需的参数
    train_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        softmax,
        EPOCHS,
        VALIDATE_EVERY,
        GRADIENT_ACCUMULATION,
        device,
        world_size,
        is_master,
        local_rank,
        PAD_TOKEN_ID,
        model_name,
        ckpt_dir
    )

if __name__ == "__main__":
    # 程序入口
    # 当脚本直接运行时，调用 main 函数开始执行
    main()
