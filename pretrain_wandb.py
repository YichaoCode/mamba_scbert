# -*- coding: utf-8 -*-
# pretrain_wandb_10_02.py

import logging
import os

os.environ["WANDB_DISABLE_SERVICE"] = "true"

import traceback
import psutil
import time

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
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import scanpy as sc
import anndata as ad
from utils import *
import wandb
import socket
from collections import Counter
from thop import profile
from torch.nn.utils import clip_grad_norm_
import time
import psutil
import torch.cuda as cuda



class DistributedTrainingUtils:
    @staticmethod
    def print_dist_init_confirmation(local_rank):
        if dist.is_initialized():
            logging.info(f"Distributed training initialized. World size: {dist.get_world_size()}, "
                         f"Rank: {dist.get_rank()}, Local Rank: {local_rank}, "
                         f"Master: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
        else:
            logging.error("Failed to initialize distributed training.")

    @staticmethod
    def record_distributed_operation(operation, *args, **kwargs):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        try:
            result = operation(*args, **kwargs)
            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)
            logging.info(f"{operation.__name__} completed in {elapsed_time:.2f}ms")
            return result
        except Exception as e:
            logging.error(f"Error during {operation.__name__}: {e}")
            raise

    @staticmethod
    def compare_model_parameters(model):
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return

        local_params = torch.cat([p.data.view(-1) for p in model.parameters()])

        if dist.get_rank() == 0:
            global_params = [torch.zeros_like(local_params) for _ in range(dist.get_world_size())]
            DistributedTrainingUtils.record_distributed_operation(dist.gather, tensor=local_params, gather_list=global_params, dst=0)

            is_consistent = True
            for i, params in enumerate(global_params):
                if not torch.equal(global_params[0], params):
                    logging.warning(f"Parameters inconsistent: Node 0 vs Node {i}")
                    is_consistent = False

            if is_consistent:
                logging.info("All nodes have consistent model parameters.")
            else:
                logging.error("Warning: Inconsistent model parameters detected between nodes.")
        else:
            DistributedTrainingUtils.record_distributed_operation(dist.gather, tensor=local_params, dst=0)

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser()
        
        # ===== 分布式训练参数 =====
        parser.add_argument("--local_rank", type=int, 
                          help='Local process rank for distributed training')
                          
        # ===== 通用训练参数 =====
        parser.add_argument("--seed", type=int, default=2021,
                          help='Random seed for reproducibility')
        parser.add_argument("--epoch", type=int, default=100,
                          help='Number of training epochs')
        parser.add_argument("--batch_size", type=int, default=4,
                          help='Training batch size')
        parser.add_argument("--learning_rate", type=float, default=1e-4,
                          help='Initial learning rate')
        parser.add_argument("--grad_acc", type=int, default=60,
                          help='Number of gradient accumulation steps')
        parser.add_argument("--valid_every", type=int, default=1,
                          help='Validate every N epochs')
                          
        # ===== 数据处理参数 =====
        parser.add_argument("--bin_num", type=int, default=5,
                          help='Number of bins for expression discretization')
        parser.add_argument("--gene_num", type=int, default=16906,
                          help='Number of genes in the dataset')
        parser.add_argument("--mask_prob", type=float, default=0.15,
                          help='Probability of masking a token')
        parser.add_argument("--replace_prob", type=float, default=0.9,
                          help='Probability of replacing a masked token with [MASK]')
                          
        # ===== 数据和模型路径 =====
        parser.add_argument("--data_path", type=str, default='./data/panglao_human.h5ad',
                          help='Path to the training data file')
        parser.add_argument("--ckpt_dir", type=str, default='./ckpts/',
                          help='Directory to save checkpoints')
        parser.add_argument("--model_name", type=str, default='panglao_pretrain',
                          help='Name of the model for saving')
                          
        # ===== 模型选择参数 =====
        parser.add_argument("--model_type", type=str, default="performer",
                          choices=["performer", "mamba"],
                          help='Type of model to use (performer or mamba)')
        parser.add_argument("--pos_embed", type=bool, default=True,
                          help='Whether to use positional embeddings')
                          
        # ===== Performer 特定参数 =====
        parser.add_argument("--performer_dim", type=int, default=200,
                          help='Hidden dimension for Performer')
        parser.add_argument("--performer_depth", type=int, default=6,
                          help='Number of layers for Performer')
        parser.add_argument("--performer_heads", type=int, default=10,
                          help='Number of attention heads for Performer')
        parser.add_argument("--performer_local_attn_heads", type=int, default=0,
                          help='Number of local attention heads')
                          
        # ===== Mamba 特定参数 =====
        # 基础结构参数
        parser.add_argument("--d_model", type=int, default=16,
                          help='Hidden dimension (d_model) for Mamba')
        parser.add_argument("--n_layer", type=int, default=4,
                          help='Number of layers (n_layer) for Mamba')
        
        # SSM相关配置
        parser.add_argument("--ssm_cfg_d_state", type=int, default=16,
                          help='SSM state size in ssm_cfg')
        parser.add_argument("--ssm_cfg_d_conv", type=int, default=4,
                          help='Conv kernel size in ssm_cfg')
        parser.add_argument("--ssm_cfg_expand", type=int, default=2,
                          help='Expansion factor in ssm_cfg')
        parser.add_argument("--ssm_cfg_dt_rank", type=str, default="auto",
                          help='Rank of Δ projection in ssm_cfg')
        parser.add_argument("--ssm_cfg_dt_min", type=float, default=0.001,
                          help='Minimum Δ value in ssm_cfg')
        parser.add_argument("--ssm_cfg_dt_max", type=float, default=0.1,
                          help='Maximum Δ value in ssm_cfg')
        parser.add_argument("--ssm_cfg_dt_init", type=str, default="random",
                          help='Initialization method for Δ in ssm_cfg')
        parser.add_argument("--ssm_cfg_dt_scale", type=float, default=1.0,
                          help='Scaling factor for Δ in ssm_cfg')
        
        # 模型训练配置
        parser.add_argument("--rms_norm", type=bool, default=True,
                          help='Whether to use RMS normalization')
        parser.add_argument("--residual_in_fp32", type=bool, default=True,
                          help='Whether to compute residuals in fp32')
        parser.add_argument("--fused_add_norm", type=bool, default=True,
                          help='Whether to use fused add norm')
        parser.add_argument("--pad_vocab_size_multiple", type=int, default=8,
                          help='Pad vocabulary size to be a multiple of this value')
                          
        self.args = parser.parse_args()

class DataModule:
    def __init__(self, args, device, world_size):
        self.args = args
        self.device = device
        self.world_size = world_size
        self.SEED = args.seed
        self.BATCH_SIZE = args.batch_size
        self.CLASS = args.bin_num + 2
        self.SEQ_LEN = args.gene_num + 1
        self.PAD_TOKEN_ID = self.CLASS - 1

    def setup(self):
        data = sc.read_h5ad(self.args.data_path)
        data = data.X
        data_train, data_val = train_test_split(data, test_size=0.05, random_state=self.SEED)

        self.train_dataset = SCDataset(data_train, self.CLASS, self.device)
        self.val_dataset = SCDataset(data_val, self.CLASS, self.device)

        self.train_sampler = DistributedSampler(self.train_dataset)
        self.val_sampler = SequentialDistributedSampler(self.val_dataset, batch_size=self.BATCH_SIZE, world_size=self.world_size)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE, sampler=self.train_sampler)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.BATCH_SIZE, sampler=self.val_sampler)

class SCDataset(Dataset):
    def __init__(self, data, CLASS, device):
        super().__init__()
        self.data = data
        self.CLASS = CLASS
        self.device = device

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (self.CLASS - 2)] = self.CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(self.device)
        return full_seq

    def __len__(self):
        return self.data.shape[0]

class Trainer:
    def __init__(self, model, optimizer, scheduler, loss_fn, train_loader, val_loader, args, device, world_size, local_rank, is_master):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_master = is_master
        self.GRADIENT_ACCUMULATION = args.grad_acc
        self.MASK_PROB = args.mask_prob
        self.REPLACE_PROB = args.replace_prob
        self.RANDOM_TOKEN_PROB = 0.
        self.MASK_TOKEN_ID = args.bin_num + 1
        self.PAD_TOKEN_ID = args.bin_num + 1
        self.MASK_IGNORE_TOKEN_IDS = [0]
        self.CLASS = args.bin_num + 2
        self.SOFTMAX = nn.Softmax(dim=-1)
        self.EPOCHS = args.epoch
        self.model_name = args.model_name
        self.ckpt_dir = args.ckpt_dir
        self.VALIDATE_EVERY = args.valid_every

    def prob_mask_like(self, t, prob):
        return torch.zeros_like(t).float().uniform_(0, 1) < prob

    def mask_with_tokens(self, t, token_ids):
        init_no_mask = torch.full_like(t, False, dtype=torch.bool)
        mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
        return mask

    def get_mask_subset_with_prob(self, mask, prob):
        batch, seq_len, device = *mask.shape, mask.device
        max_masked = math.ceil(prob * seq_len)
        num_tokens = mask.sum(dim=-1, keepdim=True)
        mask_excess = torch.cat((torch.zeros(0), torch.arange(mask.size(-1)).repeat(mask.size(0)))).reshape(mask.size(0), mask.size(-1)).to(device)
        mask_excess = (mask_excess >= (num_tokens * prob).ceil())
        mask_excess = mask_excess[:, :max_masked]
        rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
        _, sampled_indices = rand.topk(max_masked, dim=-1)
        sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)
        new_mask = torch.zeros((batch, seq_len + 1), device=device)
        new_mask.scatter_(-1, sampled_indices, 1)
        return new_mask[:, 1:].bool()

    def data_mask(self, data):
        mask_ignore_token_ids = set([*self.MASK_IGNORE_TOKEN_IDS, self.PAD_TOKEN_ID])
        no_mask = self.mask_with_tokens(data, mask_ignore_token_ids)
        mask = self.get_mask_subset_with_prob(~no_mask, self.MASK_PROB)
        masked_input = data.clone().detach()
        if self.RANDOM_TOKEN_PROB > 0:
            assert num_tokens is not None, 'num_tokens keyword must be supplied when instantiating MLM if using random token replacement'
            random_token_prob = self.prob_mask_like(data, self.RANDOM_TOKEN_PROB)
            random_tokens = torch.randint(0, num_tokens, data.shape, device=data.device)
            random_no_mask = self.mask_with_tokens(random_tokens, mask_ignore_token_ids)
            random_token_prob &= ~random_no_mask
            random_indices = torch.nonzero(random_token_prob, as_tuple=True)
            masked_input[random_indices] = random_tokens[random_indices]
        replace_prob = self.prob_mask_like(data, self.REPLACE_PROB)
        masked_input = masked_input.masked_fill(mask * replace_prob, self.MASK_TOKEN_ID)
        labels = data.masked_fill(~mask, self.PAD_TOKEN_ID)
        return masked_input, labels

    def calculate_perplexity(self, loss):
        return torch.exp(loss)

    def get_parameter_norm(self):
        total_norm = 0
        for p in self.model.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def train(self):
        """Training loop with enhanced logging and error handling."""
        dist.barrier()
        hostname = socket.gethostname()
        logging.info(f"Starting training on {hostname} with {self.args.model_type} model")
        logging.info(f"Training configuration: batch_size={self.args.batch_size}, grad_acc={self.GRADIENT_ACCUMULATION}")
        
        for i in range(1, self.EPOCHS + 1):
            epoch_start_time = time.time()
            self.train_loader.sampler.set_epoch(i)
            self.model.train()
            dist.barrier()
            
            # Initialize metrics
            running_loss = 0.0
            cum_acc = 0.0
            batch_times = []
            
            logging.info(f"=== Starting Epoch {i}/{self.EPOCHS} ===")
            
            for index, data in enumerate(self.train_loader):
                batch_start_time = time.time()
                index += 1
                
                try:
                    # Move data to device
                    data = data.to(self.device)
                    logging.info(f"Batch {index}: Input shape: {data.shape}, Device: {data.device}, Max value: {data.max().item()}, Min value: {data.min().item()}")
                    
                    # Apply masking
                    data, labels = self.data_mask(data)
                    logging.info(f"After masking - Data shape: {data.shape}, Labels shape: {labels.shape}")
                    
                    # Gradient accumulation logic
                    if index % self.GRADIENT_ACCUMULATION != 0:
                        with self.model.no_sync():
                            # Forward pass
                            if self.args.model_type == "performer":
                                logits = self.model(data)
                            elif self.args.model_type == "mamba":
                                outputs = self.model(data)
                                logits = outputs.logits
                                logging.info(f"Mamba logits shape: {logits.shape}, Max logit: {logits.max().item()}, Min logit: {logits.min().item()}")
                            
                            # Calculate loss
                            loss = self.loss_fn(logits.transpose(1, 2), labels) / self.GRADIENT_ACCUMULATION
                            
                            # Backward pass
                            loss.backward()
                    else:
                        # Forward pass with gradient accumulation step
                        if self.args.model_type == "performer":
                            logits = self.model(data)
                        elif self.args.model_type == "mamba":
                            outputs = self.model(data)
                            logits = outputs.logits
                        
                        loss = self.loss_fn(logits.transpose(1, 2), labels) / self.GRADIENT_ACCUMULATION
                        loss.backward()
                        
                        # Gradient clipping
                        grad_norm = clip_grad_norm_(self.model.parameters(), int(1e2))
                        logging.info(f"Gradient norm after clipping: {grad_norm}")
                        
                        # Optimizer step
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        
                        # Compare model parameters across processes
                        DistributedTrainingUtils.compare_model_parameters(self.model)
                    
                    # Update metrics
                    running_loss += loss.item()
                    final = self.SOFTMAX(logits)[..., 1:-1]
                    final = final.argmax(dim=-1) + 1
                    pred_num = (labels != self.PAD_TOKEN_ID).sum(dim=-1)
                    correct_num = ((labels != self.PAD_TOKEN_ID) * (final == labels)).sum(dim=-1)
                    batch_acc = torch.true_divide(correct_num, pred_num).mean().item()
                    cum_acc += batch_acc
                    
                    # Timing and resource usage
                    batch_time = time.time() - batch_start_time
                    batch_times.append(batch_time)
                    
                    # Log batch information
                    logging.info(f'Process {self.local_rank}, Epoch {i}, Step {index}, Loss: {loss.item():.6f}, Accuracy: {batch_acc:.4f}, Batch Time: {batch_time:.2f}s')
                    
                    # WandB logging for master process
                    if self.is_master and index % self.GRADIENT_ACCUMULATION == 0:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        perplexity = self.calculate_perplexity(loss)
                        param_norm = self.get_parameter_norm()
                        
                        wandb.log({
                            "train_loss": loss.item(),
                            "train_accuracy": batch_acc,
                            "learning_rate": current_lr,
                            "gradient_norm": grad_norm,
                            "parameter_norm": param_norm,
                            "perplexity": perplexity.item(),
                            "gpu_memory_allocated": cuda.memory_allocated(device=self.device) / 1e9,
                            "gpu_memory_cached": cuda.memory_reserved(device=self.device) / 1e9,
                            "batch_time": batch_time,
                            "cpu_usage": psutil.cpu_percent(),
                            "ram_usage": psutil.virtual_memory().percent,
                            "hostname": hostname
                        })
                    
                except Exception as e:
                    logging.error(f"Error in training batch {index}: {str(e)}")
                    logging.error(f"Stack trace: {traceback.format_exc()}")
                    raise
            
            # End of epoch processing
            epoch_loss = running_loss / index
            epoch_acc = 100 * cum_acc / index
            epoch_loss = get_reduced(epoch_loss, self.local_rank, 0, self.world_size)
            epoch_acc = get_reduced(epoch_acc, self.local_rank, 0, self.world_size)
            epoch_time = time.time() - epoch_start_time
            
            if self.is_master:
                avg_batch_time = sum(batch_times) / len(batch_times)
                logging.info(f'=== Epoch {i} Summary ===')
                logging.info(f'Training Loss: {epoch_loss:.6f}')
                logging.info(f'Accuracy: {epoch_acc:6.4f}%')
                logging.info(f'Average Batch Time: {avg_batch_time:.2f}s')
                logging.info(f'Total Epoch Time: {epoch_time:.2f}s')
                
                wandb.log({
                    "epoch": i,
                    "epoch_train_loss": epoch_loss,
                    "epoch_train_accuracy": epoch_acc,
                    "epoch_time": epoch_time,
                    "avg_batch_time": avg_batch_time,
                    "hostname": hostname
                })
            
            dist.barrier()
            self.scheduler.step()
            
            # Model parameter comparison and validation
            DistributedTrainingUtils.compare_model_parameters(self.model)
            
            if i % self.VALIDATE_EVERY == 0:
                self.validate(i)
                
            if self.is_master:
                self.save_checkpoint(i)
                wandb.save(os.path.join(self.ckpt_dir, f"{self.model_name}_epoch_{i}.pth"))

    def validate(self, epoch):
        """Validation loop with enhanced logging and error handling."""
        self.model.eval()
        dist.barrier()
        
        logging.info(f"=== Starting Validation for Epoch {epoch} ===")
        
        running_loss = 0.0
        predictions = []
        truths = []
        running_perplexity = 0.0
        val_batch_times = []
        hostname = socket.gethostname()
        
        try:
            with torch.no_grad():
                for index, data in enumerate(self.val_loader):
                    batch_start_time = time.time()
                    index += 1
                    
                    # Move data to device
                    data = data.to(self.device)
                    logging.info(f"Validation Batch {index}: Input shape: {data.shape}, Device: {data.device}")
                    
                    # Apply masking
                    data, labels = self.data_mask(data)
                    
                    # Forward pass
                    try:
                        if self.args.model_type == "performer":
                            logits = self.model(data)
                        elif self.args.model_type == "mamba":
                            outputs = self.model(data)
                            logits = outputs.logits
                            logging.info(f"Validation Mamba logits shape: {logits.shape}")
                        
                        loss = self.loss_fn(logits.transpose(1, 2), labels)
                        perplexity = self.calculate_perplexity(loss)
                        
                        running_loss += loss.item()
                        running_perplexity += perplexity.item()
                        
                        # Calculate predictions
                        final = self.SOFTMAX(logits)[..., 1:-1]
                        final = final.argmax(dim=-1) + 1
                        predictions.append(final)
                        truths.append(labels)
                        
                        # Timing
                        batch_time = time.time() - batch_start_time
                        val_batch_times.append(batch_time)
                        
                        logging.info(f"Validation Batch {index} - Loss: {loss.item():.6f}, Perplexity: {perplexity.item():.2f}, Time: {batch_time:.2f}s")
                        
                    except Exception as e:
                        logging.error(f"Error in validation forward pass (batch {index}): {str(e)}")
                        logging.error(f"Stack trace: {traceback.format_exc()}")
                        raise
                
                # Process validation results
                del data, labels, logits, final
                predictions = distributed_concat(torch.cat(predictions, dim=0), 
                                            len(self.val_sampler.dataset), 
                                            self.world_size)
                truths = distributed_concat(torch.cat(truths, dim=0), 
                                        len(self.val_sampler.dataset), 
                                        self.world_size)
                
                # Calculate metrics
                correct_num = ((truths != self.PAD_TOKEN_ID) * (predictions == truths)).sum(dim=-1)[0].item()
                val_num = (truths != self.PAD_TOKEN_ID).sum(dim=-1)[0].item()
                val_loss = running_loss / index
                val_loss = get_reduced(val_loss, self.local_rank, 0, self.world_size)
                avg_perplexity = running_perplexity / index
                
                if self.is_master:
                    val_acc = 100 * correct_num / val_num
                    avg_batch_time = sum(val_batch_times) / len(val_batch_times)
                    
                    logging.info(f"=== Validation Results for Epoch {epoch} ===")
                    logging.info(f"Validation Loss: {val_loss:.6f}")
                    logging.info(f"Validation Accuracy: {val_acc:6.4f}%")
                    logging.info(f"Average Perplexity: {avg_perplexity:.2f}")
                    logging.info(f"Average Batch Time: {avg_batch_time:.2f}s")
                    
                    wandb.log({
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "val_perplexity": avg_perplexity,
                        "val_batch_time": avg_batch_time,
                        "hostname": hostname
                    })
                    
                    # Log best and worst samples
                    best_sample_idx = ((truths != self.PAD_TOKEN_ID) * (predictions == truths)).sum(dim=-1).argmax()
                    worst_sample_idx = ((truths != self.PAD_TOKEN_ID) * (predictions == truths)).sum(dim=-1).argmin()
                    
                    wandb.log({
                        "best_sample": wandb.Table(
                            data=[[truths[best_sample_idx].cpu().numpy(), 
                                predictions[best_sample_idx].cpu().numpy()]], 
                            columns=["Truth", "Prediction"]
                        ),
                        "worst_sample": wandb.Table(
                            data=[[truths[worst_sample_idx].cpu().numpy(), 
                                predictions[worst_sample_idx].cpu().numpy()]], 
                            columns=["Truth", "Prediction"]
                        ),
                        "hostname": hostname
                    })
                
                del predictions, truths
                
        except Exception as e:
            logging.error(f"Error in validation: {str(e)}")
            logging.error(f"Stack trace: {traceback.format_exc()}")
            raise

    def save_checkpoint(self, epoch):
        save_ckpt(epoch, self.model, self.optimizer, self.scheduler, None, self.model_name, self.ckpt_dir)

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser()
        
        # ===== 分布式训练参数 =====
        parser.add_argument("--local_rank", type=int, 
                          help='Local process rank for distributed training')
                          
        # ===== 通用训练参数 =====
        parser.add_argument("--seed", type=int, default=2021,
                          help='Random seed for reproducibility')
        parser.add_argument("--epoch", type=int, default=100,
                          help='Number of training epochs')
        parser.add_argument("--batch_size", type=int, default=4,
                          help='Training batch size')
        parser.add_argument("--learning_rate", type=float, default=1e-4,
                          help='Initial learning rate')
        parser.add_argument("--grad_acc", type=int, default=60,
                          help='Number of gradient accumulation steps')
        parser.add_argument("--valid_every", type=int, default=1,
                          help='Validate every N epochs')
                          
        # ===== 数据处理参数 =====
        parser.add_argument("--bin_num", type=int, default=5,
                          help='Number of bins for expression discretization')
        parser.add_argument("--gene_num", type=int, default=16906,
                          help='Number of genes in the dataset')
        parser.add_argument("--mask_prob", type=float, default=0.15,
                          help='Probability of masking a token')
        parser.add_argument("--replace_prob", type=float, default=0.9,
                          help='Probability of replacing a masked token with [MASK]')
                          
        # ===== 数据和模型路径 =====
        parser.add_argument("--data_path", type=str, default='./data/panglao_human.h5ad',
                          help='Path to the training data file')
        parser.add_argument("--ckpt_dir", type=str, default='./ckpts/',
                          help='Directory to save checkpoints')
        parser.add_argument("--model_name", type=str, default='panglao_pretrain',
                          help='Name of the model for saving')
                          
        # ===== 模型选择参数 =====
        parser.add_argument("--model_type", type=str, default="performer",
                          choices=["performer", "mamba"],
                          help='Type of model to use (performer or mamba)')
        parser.add_argument("--pos_embed", type=bool, default=True,
                          help='Whether to use positional embeddings')
                          
        # ===== Performer 特定参数 =====
        parser.add_argument("--performer_dim", type=int, default=200,
                          help='Hidden dimension for Performer')
        parser.add_argument("--performer_depth", type=int, default=6,
                          help='Number of layers for Performer')
        parser.add_argument("--performer_heads", type=int, default=10,
                          help='Number of attention heads for Performer')
        parser.add_argument("--performer_local_attn_heads", type=int, default=0,
                          help='Number of local attention heads')
                          
        # ===== Mamba 特定参数 =====
        # 基础结构参数
        parser.add_argument("--d_model", type=int, default=16,
                          help='Hidden dimension (d_model) for Mamba')
        parser.add_argument("--n_layer", type=int, default=4,
                          help='Number of layers (n_layer) for Mamba')
        
        # SSM相关配置
        parser.add_argument("--ssm_cfg_d_state", type=int, default=16,
                          help='SSM state size in ssm_cfg')
        parser.add_argument("--ssm_cfg_d_conv", type=int, default=4,
                          help='Conv kernel size in ssm_cfg')
        parser.add_argument("--ssm_cfg_expand", type=int, default=2,
                          help='Expansion factor in ssm_cfg')
        parser.add_argument("--ssm_cfg_dt_rank", type=str, default="auto",
                          help='Rank of Δ projection in ssm_cfg')
        parser.add_argument("--ssm_cfg_dt_min", type=float, default=0.001,
                          help='Minimum Δ value in ssm_cfg')
        parser.add_argument("--ssm_cfg_dt_max", type=float, default=0.1,
                          help='Maximum Δ value in ssm_cfg')
        parser.add_argument("--ssm_cfg_dt_init", type=str, default="random",
                          help='Initialization method for Δ in ssm_cfg')
        parser.add_argument("--ssm_cfg_dt_scale", type=float, default=1.0,
                          help='Scaling factor for Δ in ssm_cfg')
        
        # 模型训练配置
        parser.add_argument("--rms_norm", type=bool, default=True,
                          help='Whether to use RMS normalization')
        parser.add_argument("--residual_in_fp32", type=bool, default=True,
                          help='Whether to compute residuals in fp32')
        parser.add_argument("--fused_add_norm", type=bool, default=True,
                          help='Whether to use fused add norm')
        parser.add_argument("--pad_vocab_size_multiple", type=int, default=8,
                          help='Pad vocabulary size to be a multiple of this value')
                          
        self.args = parser.parse_args()

def create_model(args, device):
    """
    创建模型实例
    Args:
        args: 配置参数
        device: 运行设备
    Returns:
        model: 创建的模型实例
    """
    logging.info(f"Creating {args.model_type} model with the following configuration:")
    
    try:
        if args.model_type == "performer":
            try:
                from performer_pytorch import PerformerLM
                
                logging.info("Performer Configuration:")
                logging.info(f"- Vocabulary size: {args.bin_num + 2}")
                logging.info(f"- Hidden dimension: {args.performer_dim}")
                logging.info(f"- Number of layers: {args.performer_depth}")
                logging.info(f"- Number of heads: {args.performer_heads}")
                logging.info(f"- Sequence length: {args.gene_num + 1}")
                logging.info(f"- Using positional embedding: {args.pos_embed}")
                
                model = PerformerLM(
                    num_tokens=args.bin_num + 2,          
                    dim=args.performer_dim,               
                    depth=args.performer_depth,           
                    max_seq_len=args.gene_num + 1,       
                    heads=args.performer_heads,           
                    local_attn_heads=args.performer_local_attn_heads,  
                    g2v_position_emb=args.pos_embed      
                )
                logging.info("Successfully created Performer model")
                
            except ImportError as e:
                logging.error(f"Failed to import Performer: {str(e)}")
                raise
                
        elif args.model_type == "mamba":
            try:
                from mamba_ssm import MambaLMHeadModel
                from mamba_ssm.models.config_mamba import MambaConfig
                
                # 确保词汇表大小是8的倍数
                vocab_size = args.bin_num + 2
                vocab_size = ((vocab_size + args.pad_vocab_size_multiple - 1) 
                            // args.pad_vocab_size_multiple 
                            * args.pad_vocab_size_multiple)
                
                # 配置SSM参数
                ssm_cfg = {
                    "d_state": args.ssm_cfg_d_state,
                    "d_conv": args.ssm_cfg_d_conv,
                    "expand": args.ssm_cfg_expand,
                    "dt_rank": args.ssm_cfg_dt_rank,
                    "dt_min": args.ssm_cfg_dt_min,
                    "dt_max": args.ssm_cfg_dt_max,
                    "dt_init": args.ssm_cfg_dt_init,
                    "dt_scale": args.ssm_cfg_dt_scale
                }
                
                # 创建配置
                config = MambaConfig(
                    d_model=args.d_model,
                    n_layer=args.n_layer,
                    vocab_size=vocab_size,
                    ssm_cfg=ssm_cfg,
                    rms_norm=args.rms_norm,
                    residual_in_fp32=args.residual_in_fp32,
                    fused_add_norm=args.fused_add_norm,
                    pad_vocab_size_multiple=args.pad_vocab_size_multiple
                )
                
                # 记录配置信息
                logging.info("Mamba Configuration:")
                logging.info(f"- Model dimension (d_model): {config.d_model}")
                logging.info(f"- Number of layers (n_layer): {config.n_layer}")
                logging.info(f"- Vocabulary size: {config.vocab_size}")
                logging.info(f"- SSM config: {config.ssm_cfg}")
                logging.info(f"- RMS norm: {config.rms_norm}")
                logging.info(f"- Residual in FP32: {config.residual_in_fp32}")
                logging.info(f"- Fused add norm: {config.fused_add_norm}")
                logging.info(f"- Vocab padding multiple: {config.pad_vocab_size_multiple}")
                
                try:
                    logging.info(f"Attempting to create Mamba model with config: {config}")
                    model = MambaLMHeadModel(config)
                    logging.info("Successfully created Mamba model")
                    
                    # 打印模型结构
                    model_structure = str(model)
                    logging.info(f"Mamba model structure:\n{model_structure}")
                    
                except Exception as e:
                    logging.error(f"Failed to create Mamba model with config: {config}")
                    logging.error(f"Error: {str(e)}")
                    raise
                    
            except ImportError as e:
                logging.error(f"Failed to import Mamba: {str(e)}")
                raise
                
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
        
        # 打印模型统计信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Model Statistics:")
        logging.info(f"- Total parameters: {total_params:,}")
        logging.info(f"- Trainable parameters: {trainable_params:,}")
        
        # 将模型移动到指定设备
        model = model.to(device)
        logging.info(f"Model moved to device: {device}")
        
        return model
        
    except Exception as e:
        logging.error(f"Error in create_model: {str(e)}")
        logging.error(f"Stack trace: {traceback.format_exc()}")
        raise

def main():
    hostname = socket.gethostname()
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - [Process %(process)d] - %(levelname)s - {hostname} - %(message)s'
    )
    config = Config()
    args = config.args

    # Initialize the distributed process group
    dist.init_process_group(backend='nccl')

    # Get local_rank and rank
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_master = rank == 0

    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Seed all processes
    seed_all(args.seed + rank)

    # Initialize WandB only in the master process
    if is_master:
        wandb.init(project="scbert-pretraining", config=args)
        script_path = os.path.abspath(__file__)
        wandb.save(script_path)

    # Print distributed initialization confirmation
    DistributedTrainingUtils.print_dist_init_confirmation(local_rank)

    # Set up data module
    data_module = DataModule(args, device, world_size)
    data_module.setup()

    # Create model and wrap with DDP
    model = create_model(args, device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Set up optimizer and scheduler
    if args.model_type == "performer":
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
    elif args.model_type == "mamba":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=15,
        T_mult=2,
        eta_min=1e-6
    )

    # Define loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=args.bin_num + 1, reduction='mean').to(device)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_loader=data_module.train_loader,
        val_loader=data_module.val_loader,
        args=args,
        device=device,
        world_size=world_size,
        local_rank=local_rank,
        is_master=is_master
    )

    # Start training
    trainer.train()

    # Finish WandB run in master process
    if is_master:
        wandb.finish()

if __name__ == "__main__":
    main()