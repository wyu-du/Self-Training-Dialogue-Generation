# -*- coding: utf-8 -*-

import torch
import json
import argparse
import os
from data_utils import parse_input_data, make_batch_inputs, prepare_aug
from decode import top_k_top_p_filtering
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.nn import functional as F
from scipy.integrate import quad
import scipy.stats as st
import numpy as np


def enable_dropout(model):
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()


def compute_scores(amr, text, model, tokenizer, loss_func, args):
    inp = f"{amr} & {text} {tokenizer.eos_token}"
    features = tokenizer(inp, padding=False, return_tensors='pt')
    features = dict([(k, v.to(args.device)) for k, v in features.items()])
    inputs = features['input_ids'] # 1 x S
    seq_len = inputs.size(1)
    
    with torch.no_grad():
        if args.num_dropouts > 1:
            enable_dropout(model)
        ppl_list = []
        for t in range(args.num_dropouts):
            outputs = model(input_ids=inputs)
            all_token_logits = outputs[0][:, :-1, :] # 1 x (S-1) x V
            pred_logp = loss_func(all_token_logits.contiguous().view(-1, model.config.vocab_size), 
                                  inputs[:, 1:].contiguous().view(-1)) # 1
            ppl_list.append(pred_logp.item())
        avg_ppl = np.mean(ppl_list)
        std_ppl = np.std(ppl_list)
        tmp = {'amr': amr, 'text': text,  
               'mean': avg_ppl, 'variability': std_ppl}
        # print(tmp)
    return tmp


def remove_outliers(ppl_list, var_list):
    new_ppl_list, new_var_list = [], []
    ppl_min = np.percentile(ppl_list, 1)
    ppl_max = np.percentile(ppl_list, 99)
    for ppl in ppl_list:
        if ppl > ppl_min and ppl < ppl_max:
            new_ppl_list.append(ppl)
    
    var_min = np.percentile(var_list, 1)
    var_max = np.percentile(var_list, 99)
    for var in var_list:
        if var > var_min and var < var_max:
            new_var_list.append(var)
    return new_ppl_list, new_var_list


def select_aug_data(args):
    select_data_path = f'{args.root_data}/{args.dataset}/{args.domain}'
    if not os.path.exists(select_data_path):
        os.makedirs(select_data_path)
    # Load original training set
    with open(f'../data/{args.dataset}/{args.domain}/train.txt', 'r') as f:
        ori_train = f.read().strip().split('\n')
    # Load augmented data
    aug_data_path = f"{args.root_data}/augmented/{args.model_name}_{args.flag}_{args.dataset}_{args.domain}_{args.train_set}_{args.timestamp}"
    with open(f"{aug_data_path}/augmented_data.json", 'r') as f:
        lines = json.load(f)
    # print('Data loaded!!!')

    # Load the model
    tokenizer = GPT2Tokenizer.from_pretrained(args.ckpt_path)
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(0)
    # print(f"Vocab size: {len(tokenizer)}")
    model = GPT2LMHeadModel.from_pretrained(args.ckpt_path)
    model = model.to(args.device)
    # print('Model loaded!!!')

    # Compute scores for original labeled data
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
    all_data, mean_list, var_list = [], [], []
    for line in ori_train:
        amr = line.split(' & ')[0]
        text = line.split(' & ')[1]
        tmp = compute_scores(amr, text, model, tokenizer, loss_func, args)
        all_data.append(tmp)
        mean_list.append(tmp['mean'])
        var_list.append(tmp['variability'])
    with open(f"{aug_data_path}/training_data_scores.json", 'w') as f:
        json.dump(all_data, f, indent=2)

    # Compute scores for augmented data
    aug_all_data, aug_mean_list, aug_var_list = [], [], []
    for line in lines:
        tmp = compute_scores(line['amr'], line['text']['text'], model, tokenizer, loss_func, args)
        aug_all_data.append(tmp)
        aug_mean_list.append(tmp['mean'])
        aug_var_list.append(tmp['variability'])
    aug_avg_mean = np.mean(aug_mean_list)
    with open(f"{aug_data_path}/augmented_data_scores.json", 'w') as f:
        json.dump(aug_all_data, f, indent=2)

    # compute threshold
    low_aug_mean_list, low_aug_var_list = [], []
    for mean, var in zip(aug_mean_list, aug_var_list):
        if mean <= aug_avg_mean:
            low_aug_mean_list.append(mean)
            low_aug_var_list.append(var)
    new_mean_list, new_var_list = remove_outliers(mean_list+low_aug_mean_list, var_list+low_aug_var_list)
    mean_threshold, _ = approximate_gaussian_mean_std(new_mean_list)
    var_threshold, _ = approximate_gaussian_mean_std(new_var_list)
    
    aug_train = []
    for line in aug_all_data:
        # select low mean and high variability data
        if line['mean'] <= mean_threshold and line['variability'] >= var_threshold:
            out_str = f"{line['amr']} & {line['text']}"
            aug_train.append(out_str.replace('\n', ' '))
    # print('SA1 Augmented training instances:', len(aug_train))
    with open(f'{select_data_path}/{args.train_set}_{args.timestamp}_{args.selection_strategy}.txt', 'w') as f:
        for pair in ori_train + aug_train:
            f.write(pair+'\n')
    sa1_data = f'{select_data_path}/{args.train_set}_{args.timestamp}_{args.selection_strategy}.txt'
    return sa1_data


def approximate_gaussian_mean_std(x):
    kde = st.gaussian_kde(x)
    # Mean and Variance - Monte Carlo
    n_samples = 10000000
    samples = kde.resample(n_samples)
    mean_mc = samples.mean()
    std_mc = samples.std()
    #print(f'Mean: {mean_mc}')
    #print(f'Standard Deviation: {std_mc}')

    low_threshold = mean_mc - std_mc
    high_threshold = mean_mc + std_mc
    #print(f'31.7% threshold: {low_threshold}')
    #print(f'68.2% threshold: {high_threshold}')
    #print('======================================\n')
    return mean_mc, std_mc


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-t', '--task', type=str, default='select', help="train/decode/augment/select")
    p.add_argument('-s', '--selection_strategy', type=str, default='sa2', help="specify the selection strategy: all/sa1/sa2")
    p.add_argument('-cp', '--ckpt_path', type=str, default=None, help="Model checkpoint")
    p.add_argument('-time', '--timestamp', type=str, default='2', help="Model checkpoint")
    p.add_argument('-f', '--flag', type=str, default='sa2', help="Model checkpoint")
    p.add_argument('-d', '--dataset', type=str, default="FewShotWoz", 
                   help="specify the dataset: FewShotWoz, FewShotSGD")
    p.add_argument('-d2', '--dataset2', type=str, default="multiwoz_amr", 
                   help="specify the dataset: multiwoz_amr")
    p.add_argument('-m', '--domain', type=str, default="restaurant", 
                   help="specify the domain: restaurant, tv, taxi, ect.")
    p.add_argument('--model_name', type=str, default="scgpt", 
                   help="specify the model name: gpt2-medium, microsoft/DialoGPT-medium")
    p.add_argument('-bz', '--batch_size', type=int, default=4)
    p.add_argument('--train_set', type=str, default="train")
    p.add_argument('--encoder_max_length', type=int, default=80)
    p.add_argument('--decoder_max_length', type=int, default=80)
    p.add_argument('--max_generation_length', type=int, default=80)
    p.add_argument('--num_sequences', type=int, default=1)
    p.add_argument('--num_dropouts', type=int, default=10)
    p.add_argument('--top_k', type=int, default=0)
    p.add_argument('--top_p', type=float, default=0.5)
    p.add_argument('--do_sample', type=bool, default=True, 
                   help="specify whether sampling from output distribution during generation")
    p.add_argument('--root_data', type=str, default="data", 
                   help="specify root dir for all generated data")
    args = p.parse_args()
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    select_aug_data(args)