# -*- coding: utf-8 -*-

import os
import re
import glob
import shutil
import argparse
import pickle
import torch

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import trange
import numpy as np


class TextSeqDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train', block_size=512, max_seq=80, seperator=' & '):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, args.model_name.split('/')[-1] + '_cached_lm_' + str(block_size) + '_seqlen_' + str(max_seq) + '_' + filename)

        print("Creating features from dataset file at %s" % directory)
        self.examples = []
        self.labels = []
        self.masks = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()      
                raw_str = line.lower()
                code_str = line.lower().split(seperator)[0] + seperator
                code_str = code_str.strip()
                if len(raw_str.split()) > max_seq -1:
                    raw_str = ' '.join(raw_str.split()[:max_seq -1])
                raw_str += ' ' + tokenizer.eos_token
                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_str))
                mask = [1] * max_seq

                if len(tokenized_text) < max_seq:
                    mask[-(max_seq - len(tokenized_text)):] = [0] * (max_seq - len(tokenized_text))
                    tokenized_text = tokenized_text + [0] * (max_seq - len(tokenized_text)) 
                else:
                    tokenized_text = tokenized_text[:max_seq]
                
                label = [-1] * max_seq
                label[:len(tokenized_text)] = tokenized_text 

                self.examples.append(tokenized_text)
                self.masks.append(mask)
                self.labels.append(label)

        print("Saving features into cached file %s" % cached_features_file)
        with open(cached_features_file, 'wb') as handle:
            pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item]), torch.tensor(self.masks[item]), torch.tensor(self.labels[item])

def load_and_cache_examples(args, tokenizer, evaluate=False):
    if int(args.timestamp) <= 1:
        train_data_file = f'../data/{args.dataset}/{args.domain}/train.txt'
    else:
        train_data_path = f'{args.root_data}/{args.dataset}/{args.domain}'
        if not os.path.exists(train_data_path):
            os.makedirs(train_data_path)
        train_data_file = f'{train_data_path}/{args.train_set}_{args.timestamp-1}_{args.selection_strategy}.txt'
    eval_data_file = f'../data/{args.dataset}/{args.domain}/dev.txt'
    dataset = TextSeqDataset(tokenizer, args, 
                             file_path=eval_data_file if evaluate else train_data_file, 
                             block_size=80, 
                             max_seq=args.max_generation_length)
    return dataset

def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= 1:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - 1)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        # print("Deleting older checkpoint [{}]".format(checkpoint))
        shutil.rmtree(checkpoint)

def train(args):
    # Load the pre-trained model
    ckpt_path = None
    if not args.finetune:
        # train from scratch
        ckpt_path = args.model_name
    else:
        ckpt_path = args.ckpt_path
    # update timestamp and create new path for ckpt
    args.output_dir = f"{args.root_models}/{args.model_name}_{args.flag}_{args.dataset}_{args.domain}_{args.train_set}_{args.timestamp}"
    
    tokenizer = GPT2Tokenizer.from_pretrained(ckpt_path)
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(0)
    model = GPT2LMHeadModel.from_pretrained(ckpt_path)
    model = model.to(args.device)
    
    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
    train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)

    t_total = len(train_dataloader) * args.epoch
    train_iterator = trange(int(args.epoch), desc="Epoch", disable=args.local_rank not in [-1, 0])
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)
    # Release all the GPU memory cache that can be freed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    global_step = 0
    tr_loss, logging_loss = 0., 0.
    eval_ppl = 100000.
    model.zero_grad()
    for e in train_iterator:
        for step, batch in enumerate(train_dataloader):
            inputs, masks, labels = batch
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            
            model.train()
            outputs = model(inputs, labels=labels)
            loss = outputs[0]
            loss = loss.mean()
            
            loss.backward()
            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
            
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                results = evaluate(args, model, tokenizer, eval_dataloader)
                cur_train_loss = (tr_loss - logging_loss)/args.logging_steps
                print(f"Global Step: {global_step}, LR: {scheduler.get_lr()[0]}, Train Loss: {cur_train_loss}, Eval PPL: {results['perplexity']}")
                logging_loss = tr_loss
                if results['perplexity'] < eval_ppl:
                    eval_ppl = results['perplexity']
                    output_dir=f"{args.output_dir}/checkpoint-{global_step}"
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    # print("Saving model checkpoint to %s" % output_dir)
                    _rotate_checkpoints(args, 'checkpoint')

    output_dir=f"{args.output_dir}/checkpoint-{global_step}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Saving model checkpoint to %s" % output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    _rotate_checkpoints(args, 'checkpoint')
    return output_dir

def evaluate(args, model, tokenizer, eval_dataloader):
    eval_output_dir = f"{args.root_models}/{args.model_name}_{args.flag}_{args.dataset}_{args.domain}_{args.train_set}_{args.timestamp}"
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    model.eval()
    eval_loss = 0.0
    nb_eval_steps = 0
    ppl_list = []
    for batch in eval_dataloader:
        inputs, masks, labels = batch
        inputs = inputs.to(args.device)
        masks = masks.to(args.device)
        labels = labels.to(args.device)
        
        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
            ppl_list.append(np.exp(lm_loss.item()))
        nb_eval_steps += 1
    
    eval_loss = eval_loss/nb_eval_steps
    ppl = torch.exp(torch.tensor(eval_loss))
    
    result = {"perplexity": ppl.item()}
    return result



if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-t', '--task', type=str, default='train', help="train/decode")
    p.add_argument('-ft', '--finetune', type=bool, default=False, help="specify whether to finetune the model")
    p.add_argument('-e', '--epoch', type=int, default=10, help="Number of training epoch per self-training iter")
    p.add_argument('-lr', '--learning_rate', type=float, default=1e-5, help="learning rate per self-training iter")
    p.add_argument('-cp', '--ckpt_path', type=str, default=None, help="Model checkpoint")
    p.add_argument('-time', '--timestamp', type=str, default='1', help="Model checkpoint")
    p.add_argument('-f', '--flag', type=str, default='base', help="Model checkpoint")
    p.add_argument('-d', '--dataset', type=str, default="FewShotWoz", 
                    help="specify the mr2text dataset: FewShotWoz, FewShotSGD")
    p.add_argument('-d2', '--dataset2', type=str, default="multiwoz_amr", 
                    help="specify the unlabeled mr dataset: multiwoz_amr, sgd_amr")
    p.add_argument('-m', '--domain', type=str, default="restaurant", 
                   help="specify the domain: restaurant, tv, taxi, ect.")
    p.add_argument('--model_name', type=str, default="scgpt", 
                   help="specify the model name: scgpt, gpt2-medium, microsoft/DialoGPT-medium")
    p.add_argument('-bz', '--batch_size', type=int, default=1)
    p.add_argument('--train_set', type=str, default="train")
    p.add_argument('--logging_steps', type=int, default=10)
    p.add_argument('--save_steps', type=int, default=100)
    p.add_argument('--encoder_max_length', type=int, default=80)
    p.add_argument('--decoder_max_length', type=int, default=80)
    p.add_argument('--max_generation_length', type=int, default=80)
    p.add_argument('--local_rank', type=int, default=-1, help="Multiple GPU training")
    p.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    p.add_argument('--root_data', type=str, default="data", help="specify root dir for all generated data")
    p.add_argument('--root_models', type=str, default="models", help="specify root dir for all model ckpts")

    args = p.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    
    if os.path.exists(f"{args.root_models}/{args.model_name}_{args.flag}_{args.dataset}_{args.domain}_{args.train_set}_{args.timestamp}"):
        shutil.rmtree(f"{args.root_models}/{args.model_name}_{args.flag}_{args.dataset}_{args.domain}_{args.train_set}_{args.timestamp}")
    train(args)
