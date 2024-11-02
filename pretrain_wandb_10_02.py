# -*- coding: utf-8 -*-

import logging
import os

os.environ["WANDB_DISABLE_SERVICE"] = "true"


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
from performer_pytorch import PerformerLM
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

# Add Mamba imports
from mamba_ssm import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig

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
        parser.add_argument("--local_rank", type=int, help='Local process rank.')
        parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
        parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
        parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
        parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
        parser.add_argument("--batch_size", type=int, default=1, help='Number of batch size.')
        parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
        parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
        parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
        parser.add_argument("--mask_prob", type=float, default=0.15, help='Probability of masking.')
        parser.add_argument("--replace_prob", type=float, default=0.9, help='Probability of replacing with [MASK] token for masking.')
        parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
        parser.add_argument("--data_path", type=str, default='./data/panglao_human.h5ad', help='Path of data for pretraining.')
        parser.add_argument("--ckpt_dir", type=str, default='./ckpts/', help='Directory of checkpoint to save.')
        parser.add_argument("--model_name", type=str, default='panglao_pretrain', help='Pretrained model name.')
        parser.add_argument("--model_type", type=str, default="performer", choices=["performer", "mamba"], help='Type of model to use')
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
        dist.barrier()
        hostname = socket.gethostname()
        for i in range(1, self.EPOCHS + 1):
            epoch_start_time = time.time()
            self.train_loader.sampler.set_epoch(i)
            self.model.train()
            dist.barrier()
            running_loss = 0.0
            cum_acc = 0.0
            for index, data in enumerate(self.train_loader):
                index += 1
                data = data.to(self.device)
                data, labels = self.data_mask(data)
                if index % self.GRADIENT_ACCUMULATION != 0:
                    with self.model.no_sync():
                        if self.args.model_type == "performer":
                            logits = self.model(data)
                        elif self.args.model_type == "mamba":
                            logits = self.model(data).logits
                        loss = self.loss_fn(logits.transpose(1, 2), labels) / self.GRADIENT_ACCUMULATION
                        loss.backward()
                if index % self.GRADIENT_ACCUMULATION == 0:
                    if self.args.model_type == "performer":
                        logits = self.model(data)
                    elif self.args.model_type == "mamba":
                        logits = self.model(data).logits
                    loss = self.loss_fn(logits.transpose(1, 2), labels) / self.GRADIENT_ACCUMULATION
                    loss.backward()
                    grad_norm = clip_grad_norm_(self.model.parameters(), int(1e2))
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    DistributedTrainingUtils.compare_model_parameters(self.model)
                
                running_loss += loss.item()
                final = self.SOFTMAX(logits)[..., 1:-1]
                final = final.argmax(dim=-1) + 1
                pred_num = (labels != self.PAD_TOKEN_ID).sum(dim=-1)
                correct_num = ((labels != self.PAD_TOKEN_ID) * (final == labels)).sum(dim=-1)
                cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
                logging.info(f'Process {self.local_rank}, Epoch {i}, Step {index}, Loss: {loss.item():.6f}')
                
                if self.is_master and index % self.GRADIENT_ACCUMULATION == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    perplexity = self.calculate_perplexity(loss)
                    param_norm = self.get_parameter_norm()
                    wandb.log({
                        "train_loss": loss.item(),
                        "train_accuracy": torch.true_divide(correct_num, pred_num).mean().item(),
                        "learning_rate": current_lr,
                        "gradient_norm": grad_norm,
                        "parameter_norm": param_norm,
                        "perplexity": perplexity.item(),
                        "gpu_memory_allocated": cuda.memory_allocated(device=self.device) / 1e9,
                        "gpu_memory_cached": cuda.memory_reserved(device=self.device) / 1e9,
                        "cpu_usage": psutil.cpu_percent(),
                        "ram_usage": psutil.virtual_memory().percent,
                        "hostname": hostname
                    })
            epoch_loss = running_loss / index
            epoch_acc = 100 * cum_acc / index
            epoch_loss = get_reduced(epoch_loss, self.local_rank, 0, self.world_size)
            epoch_acc = get_reduced(epoch_acc, self.local_rank, 0, self.world_size)
            epoch_time = time.time() - epoch_start_time
            if self.is_master:
                print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  ==')
                logging.info(f'Process {self.local_rank}, Epoch {i} Completed, Training Loss: {epoch_loss:.6f}, Accuracy: {epoch_acc:6.4f}%')
                wandb.log({
                    "epoch": i,
                    "epoch_train_loss": epoch_loss,
                    "epoch_train_accuracy": epoch_acc,
                    "epoch_time": epoch_time,
                    "hostname": hostname
                })
            dist.barrier()
            self.scheduler.step()
            
            DistributedTrainingUtils.compare_model_parameters(self.model)
            
            if i % self.VALIDATE_EVERY == 0:
                self.validate(i)
            if self.is_master:
                self.save_checkpoint(i)
                wandb.save(os.path.join(self.ckpt_dir, f"{self.model_name}_epoch_{i}.pth"))

    def validate(self, epoch):
        self.model.eval()
        dist.barrier()
        running_loss = 0.0
        predictions = []
        truths = []
        running_perplexity = 0.0
        hostname = socket.gethostname()
        with torch.no_grad():
            for index, data in enumerate(self.val_loader):
                index += 1
                data = data.to(self.device)
                data, labels = self.data_mask(data)
                if self.args.model_type == "performer":
                    logits = self.model(data)
                elif self.args.model_type == "mamba":
                    logits = self.model(data).logits
                loss = self.loss_fn(logits.transpose(1, 2), labels)
                running_loss += loss.item()
                running_perplexity += self.calculate_perplexity(loss).item()
                final = self.SOFTMAX(logits)[..., 1:-1]
                final = final.argmax(dim=-1) + 1
                predictions.append(final)
                truths.append(labels)
            del data, labels, logits, final
            predictions = distributed_concat(torch.cat(predictions, dim=0), len(self.val_sampler.dataset), self.world_size)
            truths = distributed_concat(torch.cat(truths, dim=0), len(self.val_sampler.dataset), self.world_size)
            correct_num = ((truths != self.PAD_TOKEN_ID) * (predictions == truths)).sum(dim=-1)[0].item()
            val_num = (truths != self.PAD_TOKEN_ID).sum(dim=-1)[0].item()
            val_loss = running_loss / index
            val_loss = get_reduced(val_loss, self.local_rank, 0, self.world_size)
        if self.is_master:
            val_acc = 100 * correct_num / val_num
            logging.info(f'Process {self.local_rank}, Epoch {epoch}, Validation Loss: {val_loss:.6f}, Accuracy: {val_acc:6.4f}%')
            wandb.log({
                "epoch": epoch,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "hostname": hostname
            })
            best_sample_idx = ((truths != self.PAD_TOKEN_ID) * (predictions == truths)).sum(dim=-1).argmax()
            worst_sample_idx = ((truths != self.PAD_TOKEN_ID) * (predictions == truths)).sum(dim=-1).argmin()
            wandb.log({
                "best_sample": wandb.Table(data=[[truths[best_sample_idx].cpu().numpy(), predictions[best_sample_idx].cpu().numpy()]], columns=["Truth", "Prediction"]),
                "worst_sample": wandb.Table(data=[[truths[worst_sample_idx].cpu().numpy(), predictions[worst_sample_idx].cpu().numpy()]], columns=["Truth", "Prediction"]),
                "hostname": hostname
            })
        del predictions, truths

    def save_checkpoint(self, epoch):
        save_ckpt(epoch, self.model, self.optimizer, self.scheduler, None, self.model_name, self.ckpt_dir)

def create_model(args, device):
    if args.model_type == "performer":
        model = PerformerLM(
            num_tokens=args.bin_num + 2,
            dim=200,
            depth=6,
            max_seq_len=args.gene_num + 1,
            heads=10,
            local_attn_heads=0,
            g2v_position_emb=args.pos_embed
        )
    elif args.model_type == "mamba":
        config = MambaConfig()
        config.d_model = 16
        config.n_layer = 4
        config.vocab_size = args.gene_num + 2
        config.residual_in_fp32 = False
        model = MambaLMHeadModel(config)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    return model.to(device)

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