# -*- coding: utf-8 -*-

import torch
import json
import argparse
import os
from data_utils import parse_input_data, make_batch_inputs, prepare_aug
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.nn import functional as F
from decode import top_k_top_p_filtering
import numpy as np



def generate_sentences(batch, model, tokenizer, args):
    features = make_batch_inputs(batch, tokenizer, args, args.device)
    generated_sentences = []
    for i, inputs in enumerate(features['input_ids']):
        context = inputs.unsqueeze(0)
        context_len = int(torch.sum(features['attention_mask'][i]).item())
        generated = context[:, :context_len]
        with torch.no_grad():
            for _ in range(args.max_generation_length):
                outputs = model(input_ids=generated)
                next_token_logits = outputs[0][:, -1, :] # 1 x V
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.top_k, top_p=args.top_p)
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


def augment(args):
    te_df = parse_input_data(in_file=f'../data/{args.dataset2}', domain=args.domain)
    if len(te_df) > 10000:
        te_df = te_df.iloc[:10000,:]
    print(f'Total unlabeled MRs: {len(te_df)}')
    
    # Load the model
    tokenizer = GPT2Tokenizer.from_pretrained(args.ckpt_path)
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(0)
    # print(f"Vocab size: {len(tokenizer)}")
    model = GPT2LMHeadModel.from_pretrained(args.ckpt_path)
    model = model.to(args.device)
    # print('Model loaded!!!')
    
    # Make predictions
    test_output = Dataset.from_pandas(te_df).map(
        lambda batch: {'generated': generate_sentences(
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
    with open(f"{aug_data_path}/augmented_data.json", 'w') as f:
        f.write(json.dumps(pred_list, indent=2))
    with open(f'{args.root_data}/{args.dataset}/{args.domain}/{args.train_set}_{args.timestamp}_all.txt', 'w') as f:
        for pair in pred_list:
            out_str = f"{pair['amr']} & {pair['text']['text']}"
            f.write(out_str+'\n')
    aug_data = f"{args.root_data}/{args.dataset}/{args.domain}/{args.train_set}_{args.timestamp}_all.txt"
    return aug_data



if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-t', '--task', type=str, default='augment', help="train/decode/augment/select")
    p.add_argument('-s', '--selection_strategy', type=str, default='all', help="specify the selection strategy: all/sa2")
    p.add_argument('-cp', '--ckpt_path', type=str, default=None, help="Model checkpoint")
    p.add_argument('-time', '--timestamp', type=str, default='1', help="Model checkpoint")
    p.add_argument('-f', '--flag', type=str, default='sa2', help="Model checkpoint")
    p.add_argument('-d', '--dataset', type=str, default="FewShotWoz", 
                   help="specify the dataset: FewShotWoz, FewShotSGD")
    p.add_argument('-d2', '--dataset2', type=str, default="multiwoz_amr", 
                   help="specify the dataset: multiwoz_amr, sgd_amr")
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
    p.add_argument('--do_sample', type=bool, default=True, 
                   help="specify whether sampling from output distribution during generation")
    p.add_argument('--root_data', type=str, default="data", 
                   help="specify root dir for all generated data")
    args = p.parse_args()
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    augment(args)