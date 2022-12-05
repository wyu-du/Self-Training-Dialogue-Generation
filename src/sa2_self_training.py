# -*- coding: utf-8 -*-
import os
import torch
import glob
import argparse
import random
import time
import json
import numpy as np

from train import train
from decode import decode
from woz_evaluator import woz_evaluator
from sgd_evaluator import sgd_evaluator
from augment import augment
from select_data import select_aug_data
from sa2_augment import sa2_augment
from woz_checker import woz_checker



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        

def self_training(args):
    print(f'Current iteration: {args.timestamp}')
    if args.timestamp == 1:
        args.finetune = False
        args.batch_size = 1
        args.learning_rate = 1e-5
        args.ckpt_path = train(args)
    else:
        time.sleep(3)
        args.finetune = True
        args.batch_size = 4
        # load the previous model path
        output_dir = f"{args.root_models}/{args.model_name}_{args.flag}_{args.dataset}_{args.domain}_{args.train_set}_{args.timestamp-1}"
        args.ckpt_path = glob.glob(f'{output_dir}/*/')[0] # get previous checkpoint
        args.ckpt_path = train(args) # get current checkpoint
    output_dir = f"{args.root_models}/{args.model_name}_{args.flag}_{args.dataset}_{args.domain}_{args.train_set}_{args.timestamp}"
    args.ckpt_path = glob.glob(f'{output_dir}/*/')[0] # get current checkpoint
    print(f"Current checkpoint: {args.ckpt_path}")
    # predict on dev set (use current model)
    args.task = 'decode_dev'
    args.target_file = decode(args)
    args.task = 'decode_dev'
    if args.dataset == 'FewShotWoz':
        dev_bleu, dev_err = woz_evaluator(args)
        print({'dev_bleu': dev_bleu, 'dev_err': dev_err})
    else:
        dev_bleu = sgd_evaluator(args)
        print({'dev_bleu': dev_bleu})
    # predict on test set (use current model)
    args.task = 'decode_test'
    args.target_file = decode(args)
    args.task = 'decode_test'
    if args.dataset == 'FewShotWoz':
        bleu, err = woz_evaluator(args)
        print({'test_bleu': bleu, 'test_err': err})
    else:
        bleu = sgd_evaluator(args)
        print({'test_bleu': bleu})
    # data augmentation (use current model)
    print('Start synthetic text annotation ...')
    args.selection_strategy = 'all'
    aug_data = augment(args)
    print(f'Save all augmented data to: {aug_data}')
    # data selection (use current model)
    print('Start uncertainty-based data selection ...')
    args.selection_strategy = 'sa1'
    sa1_data = select_aug_data(args)
    print(f'Save sa1 augmented data to: {sa1_data}')
    # response refinement
    print('Start refined response generation ...')
    args.selection_strategy = 'sa2'
    sa2_data = sa2_augment(args)
    print(f'Save sa2 augmented data to: {sa2_data}')



if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-t', '--task', type=str, default='train', help="train/decode/augment/select")
    p.add_argument('-s', '--selection_strategy', type=str, default='sa2', 
                   help="specify the selection strategy: all/sa1/sa2")
    p.add_argument('-ft', '--finetune', type=bool, default=False, help="specify whether to finetune the model")
    p.add_argument('-e', '--epoch', type=int, default=10, help="number of training epoch per self-training iter")
    p.add_argument('-lr', '--learning_rate', type=float, default=1e-5, help="Number of training epoch per self-training iter")
    p.add_argument('-cp', '--ckpt_path', type=str, default=None, help="model checkpoint")
    p.add_argument('-time', '--timestamp', type=int, default=1, help="model checkpoint")
    p.add_argument('-f', '--flag', type=str, default='base', help="model checkpoint")
    p.add_argument('-d', '--dataset', type=str, default="FewShotWoz", 
                   help="specify the mr2text dataset: FewShotWoz, FewShotSGD")
    p.add_argument('-d2', '--dataset2', type=str, default="multiwoz_amr", 
                   help="specify the unlabeled mr dataset: multiwoz_amr, sgd_amr")
    p.add_argument('-m', '--domain', type=str, default="restaurant", 
                   help="specify the domain: restaurant, tv, taxi, ect.")
    p.add_argument('--model_name', type=str, default="scgpt", 
                   help="specify the model name: scgpt, gpt2-medium, microsoft/DialoGPT-medium")
    p.add_argument('-bz', '--batch_size', type=int, default=4)
    p.add_argument('--logging_steps', type=int, default=10)
    p.add_argument('--save_steps', type=int, default=100)
    p.add_argument('--train_set', type=str, default="train")
    p.add_argument('--encoder_max_length', type=int, default=80)
    p.add_argument('--decoder_max_length', type=int, default=80)
    p.add_argument('--max_generation_length', type=int, default=80)
    p.add_argument('--num_sequences', type=int, default=5)
    p.add_argument('--num_dropouts', type=int, default=10)
    p.add_argument('--top_k', type=int, default=0)
    p.add_argument('--top_p', type=float, default=1.0)
    p.add_argument('--do_sample', type=bool, default=True, 
                   help="specify whether sampling from output distribution during generation")
    p.add_argument('--local_rank', type=int, default=-1, help="multiple GPU training")
    p.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    p.add_argument('--root_data', type=str, default="../data", 
                   help="specify root dir for all generated data")
    p.add_argument('--root_models', type=str, default="../models", 
                   help="specify root dir for all model ckpts")
    p.add_argument('--filter_out', action='store_true',
                   help="specify whether to filter out invalid augmented data")

    args = p.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    
    set_seed(args)
    self_training(args)
