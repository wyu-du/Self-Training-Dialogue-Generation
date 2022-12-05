# -*- coding: utf-8 -*-

import torch
import pandas as pd
from datasets import Dataset


def linear_da(cxt_intents):
    out = []
    for cxt_intent in cxt_intents:
        if type(cxt_intent) is list:
            out.append(' '.join(cxt_intent))
        else:
            out.append(cxt_intent)
    return out

def parse_data(in_file='../data/FewShotWoz/restaurant', mode='train'):
    with open(f'{in_file}/{mode}.txt', 'r') as f:
        data = f.read().strip().split('\n')
    contexted = []
    for i, line in enumerate(data):
        items = line.split(' & ')
        row = (i, items[0], items[1])
        contexted.append(row)
    columns = ['id', 'amr', 'text']
    data_df = pd.DataFrame.from_records(contexted, columns=columns)
    return data_df

def parse_input_data(in_file='../data/FewShotWoz', domain='restaurant'):
    with open(f'{in_file}/{domain}.txt', 'r') as f:
        data = f.read().strip().split('\n')
    contexted = []
    for i, line in enumerate(data):
        row = (i, line.strip())
        contexted.append(row)
    columns = ['id', 'amr']
    data_df = pd.DataFrame.from_records(contexted, columns=columns)
    return data_df

# Specific to dataset.
def construct_input_for_batch(tokenizer, batch, args):
    source, target = [], []
    if args.dataset2 in ["multiwoz_amr", "sgd_amr"]:
        # inference stage
        for amr in batch['amr']:
            inp = f"{amr} & "
            source.append(inp.strip())
            target.append(inp.strip())
        return source, target
    if args.dataset in ["FewShotWoz", "FewShotSGD"]:
        if args.task == 'train':
            # training stage
            for i in range(len(batch['id'])):
                inp = ""
                inp += f"{batch['amr'][i]} & "
                inp += f"{batch['text'][i]} {tokenizer.eos_token} "
                source.append(inp.strip())
                target.append(inp.strip())
        else:
            # inference stage
            for amr in batch['amr']:
                inp = f"{amr} & "
                source.append(inp.strip())
            
            for text in batch['text']:
                out = f"{text} {tokenizer.eos_token} "
                target.append(out.strip())
        return source, target

def make_batch_inputs(batch, tokenizer, args, device='cuda:0'):
    # Concatenate the concept names for each example in the batch.
    input_lists, _ = construct_input_for_batch(tokenizer, batch, args)
    # Use the model's tokenizer to create the batch input_ids.
    batch_features = tokenizer(input_lists, padding=True, return_tensors='pt')
    # Move all inputs to the device.
    batch_features = dict([(k, v.to(device)) for k, v in batch_features.items()])
    return batch_features

def make_batch_data(batch, tokenizer, args, device='cuda:0'):
    # Concatenate the concept names for each example in the batch.
    input_lists, label_list = construct_input_for_batch(tokenizer, batch, args)
    # Use the model's tokenizer to create the batch input_ids.
    batch_features = tokenizer(input_lists, padding=True, return_tensors='pt')
    batch_labels = tokenizer(label_list, padding=True, return_tensors='pt')
    # Move all inputs to the device.
    batch_features = dict([(k, v.to(device)) for k, v in batch_features.items()])
    batch_labels = dict([(k, v.to(device)) for k, v in batch_labels.items()])
    return batch_features, batch_labels

def batch_tokenize(dataset_batch, tokenizer, args):
    source, target = construct_input_for_batch(tokenizer, dataset_batch, args)
    res = {
          "input_ids": tokenizer(
              source,
              padding='max_length', 
              truncation=True,
              max_length=args.encoder_max_length, 
          )["input_ids"],
         "attention_mask": tokenizer(
              source,
              padding='max_length', 
              truncation=True,
              max_length=args.encoder_max_length,
          )["attention_mask"],
          "labels": tokenizer(
              target,
              padding='max_length', 
              truncation=True,
              max_length=args.encoder_max_length,
          )["input_ids"],
    }
    return res

def batchify_data(df, tokenizer, args):
    dataset = Dataset.from_pandas(df)
    data_tokenized = dataset.map(
            lambda batch: batch_tokenize(batch, tokenizer, args),
            batched=True)
    return data_tokenized

def compute_loss_gpt2(batch, model, tokenizer, args):
    # Concatenate the concept names for each example in the batch.
    args.task = 'train'
    input_lists, label_lists = construct_input_for_batch(tokenizer, batch, args)
    # Use the model's tokenizer to create the batch input_ids.
    batch_features = tokenizer(input_lists, padding=True, return_tensors='pt')
    batch_labels = tokenizer(label_lists, padding=True, return_tensors='pt')
    # Move all inputs to the device.
    device = args.device
    batch_features = dict([(k, v.to(device)) for k, v in batch_features.items()])
    batch_labels = dict([(k, v.to(device)) for k, v in batch_labels.items()])
    with torch.no_grad():
        # Note that the labels are shifted inside the model, so set labels = input_ids
        outputs = model(input_ids=batch_features['input_ids'],
                    labels=batch_features['input_ids'],
                    attention_mask=batch_features['attention_mask'])
        eval_loss = outputs.loss.item()
    return [eval_loss] 

def test_ppl_gpt2(val_df, model, tokenizer, args):
    loss_dict = Dataset.from_pandas(val_df).map(
            lambda batch: {'loss': compute_loss_gpt2(batch, model, tokenizer, args)},
            batched=True,
            batch_size=1)
  
    eval_loss = 0.
    nb_eval_steps = 0
    for item in list(loss_dict):
        eval_loss += item['loss']
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    ppl = torch.exp(torch.tensor(eval_loss))
    return ppl.item()

def prepare_eval(output_list):
    pred_list = []
    for item in output_list:
        pred_list.append(item['generated'])
    return pred_list

def prepare_aug(output_list):
    pred_list = []
    for item in output_list:
        pred_list.append({
                "amr": item['amr'],
                "text": item['generated']
                })
    return pred_list
