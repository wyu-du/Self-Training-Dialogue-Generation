# -*- coding: utf-8 -*-

import torch
import json
import argparse
import os
from data_utils import parse_input_data, make_batch_inputs, prepare_aug
from decode import top_k_top_p_filtering
from woz_checker import woz_checker
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.nn import functional as F
import numpy as np
import pandas as pd


def enable_dropout(model):
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()


def generate_sa2_sentences(batch, model, tokenizer, args):
    features = make_batch_inputs(batch, tokenizer, args, args.device)
    generated_sentences = []
    for i, inputs in enumerate(features['input_ids']):
        context = inputs.unsqueeze(0)
        context_len = int(torch.sum(features['attention_mask'][i]).item())
        generated = context[:, :context_len]
        with torch.no_grad():
            for _ in range(args.max_generation_length):
                batch_generated_logits = []
                for _ in range(args.num_dropouts):
                    enable_dropout(model)
                    outputs = model(input_ids=generated)
                    next_token_logits = outputs[0][:, -1, :] # 1 x V
                    batch_generated_logits.append(next_token_logits)
                batch_generated_logits = torch.cat(batch_generated_logits, dim=0) # T x V
                mean_generated_logits = torch.mean(batch_generated_logits, dim=0, keepdim=True) # 1 x V
                filtered_logits = top_k_top_p_filtering(mean_generated_logits, top_k=args.top_k, top_p=args.top_p)
                if args.do_sample: 
                    probs = F.softmax(filtered_logits, dim=-1) # 1 x V
                    next_token = torch.multinomial(probs, num_samples=1) # 1 x 1
                    next_token_prob = probs[0,next_token[0,0]].item()
                else:
                    next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
                    probs = F.softmax(filtered_logits, dim=-1)
                    next_token_prob = probs[0,next_token[0,0]].item()
                generated = torch.cat((generated, next_token), dim=1)
        out = generated[0, context_len:].tolist()
        text = tokenizer.decode(out, skip_special_tokens=True).strip()
        tmp = {'text': text}
        generated_sentences.append(tmp)
    return generated_sentences


def sa2_augment(args):
    ori_label_data_path = f"../data/{args.dataset}/{args.domain}/train.txt"
    old_aug_data_path = f"{args.root_data}/{args.dataset}/{args.domain}/{args.train_set}_{args.timestamp}_sa1.txt"
    with open(ori_label_data_path, 'r') as f:
        ori_label_data = f.read().strip().split('\n')
    with open(old_aug_data_path, 'r') as f:
        lines = f.read().strip().split('\n')
        old_aug_data = lines[len(ori_label_data):]

    contexted = []
    for i, line in enumerate(old_aug_data):
        amr = line.split(' & ')[0].strip()
        contexted.append((i, amr))
    te_df = pd.DataFrame.from_records(contexted, columns=['id', 'amr'])
    # print(f'Total examples: {len(te_df)}')
    
    # Load the model
    tokenizer = GPT2Tokenizer.from_pretrained(args.ckpt_path)
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(0)
    # print(f"Vocab size: {len(tokenizer)}")
    model = GPT2LMHeadModel.from_pretrained(args.ckpt_path)
    model = model.to(args.device)
    # print('Model loaded!!!')
    
    # Make predictions
    test_output = Dataset.from_pandas(te_df).map(
        lambda batch: {'generated': generate_sa2_sentences(
            batch,
            model, 
            tokenizer, 
            args)
        },
        batched=True, 
        batch_size=args.batch_size,
    )
    
    # prepare augmented data
    pred_list = prepare_aug(list(test_output))
    aug_data_path = f"{args.root_data}/augmented/{args.model_name}_{args.flag}_{args.dataset}_{args.domain}_{args.train_set}_{args.timestamp}"
    if not os.path.exists(aug_data_path):
        os.makedirs(aug_data_path)
    with open(f"{aug_data_path}/sa2_augmented_data.json", 'w') as f:
        f.write(json.dumps(pred_list, indent=2))

    new_aug_data = []
    if args.filter_out:
        new_aug_data = woz_checker(args)
    else:
        for item in pred_list:
            line = item['amr'] + ' & ' + item['text']['text']
            new_aug_data.append(line)

    select_data_path = f'{args.root_data}/{args.dataset}/{args.domain}'
    if not os.path.exists(select_data_path):
        os.makedirs(select_data_path)
    with open(f'{select_data_path}/{args.train_set}_{args.timestamp}_{args.selection_strategy}.txt', 'w') as f:
        for pair in ori_label_data + new_aug_data:
            f.write(pair+'\n')
    sa2_data = f'{select_data_path}/{args.train_set}_{args.timestamp}_{args.selection_strategy}.txt'
    return sa2_data



if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-t', '--task', type=str, default='augment', help="train/decode/augment/select")
    p.add_argument('-s', '--selection_strategy', type=str, default='sa2', help="specify the selection strategy: all/sa1/sa2")
    p.add_argument('-cp', '--ckpt_path', type=str, default=None, help="Model checkpoint")
    p.add_argument('-time', '--timestamp', type=str, default='1', help="Model checkpoint")
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
    p.add_argument('--top_p', type=float, default=0.95)
    p.add_argument('--filter_out', action='store_true',
                   help="specify whether to filter out invalid augmented data")
    p.add_argument('--do_sample', type=bool, default=True, 
                   help="specify whether sampling from output distribution during generation")
    p.add_argument('--root_data', type=str, default="data", 
                   help="specify root dir for all generated data")
    args = p.parse_args()
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sa2_augment(args)