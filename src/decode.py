# -*- coding: utf-8 -*-

import torch
import json
import argparse
import os
import glob
from torch.nn import functional as F
from data_utils import parse_data, prepare_eval, test_ppl_gpt2, make_batch_data
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequences(batch, model, tokenizer, args):
    features, labels = make_batch_data(batch, tokenizer, args, args.device)
    output_tests = []
    for i, inputs in enumerate(features['input_ids']):
        context = inputs.unsqueeze(0).repeat(args.num_sequences, 1)
        context_len = int(torch.sum(features['attention_mask'][i]).item())
        generated = context[:, :context_len]
        with torch.no_grad():
            for _ in range(args.max_generation_length):
                outputs = model(input_ids=generated)
                next_token_logits = outputs[0][:, -1, :]
                
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.top_k, top_p=args.top_p)
                if args.do_sample: 
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                else:
                    next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
                generated = torch.cat((generated, next_token), dim=1)
        out = generated[:, context_len:].tolist()
        examples = []
        for o in out:
            text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
            examples.append(text)
        output_tests.append(examples)
    return output_tests


def decode(args):
    if args.task == 'decode_test':
        mode = 'test'
    else:
        mode = 'dev'
    te_df = parse_data(in_file=f'../data/{args.dataset}/{args.domain}', mode=mode)
    # print(f'Testing examples: {len(te_df)}')
    
    # Load the model
    tokenizer = GPT2Tokenizer.from_pretrained(args.ckpt_path)
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(0)
    # print(f"Vocab size: {len(tokenizer)}")
    model = GPT2LMHeadModel.from_pretrained(args.ckpt_path)
    model = model.to(args.device)
    # print('Model loaded!!!')
    
    # Make predictions
    test_output = Dataset.from_pandas(te_df).map(
        lambda batch: {'generated': sample_sequences(
            batch,
            model, 
            tokenizer, 
            args)
        },
        batched=True, 
        batch_size=args.batch_size,
    )
    
    # # Compute ppl
    # ppl = test_ppl_gpt2(te_df, model, tokenizer, args)
    # ppl = round(ppl, 4)
    # print(f"Test ppl: {ppl}")
      
    # prepare evaluation data
    pred_list = prepare_eval(list(test_output))
    model_path = args.ckpt_path.split('/')[-3]
    target_file = f"{args.root_models}/{model_path}/{mode}_preds.json"
    print(f'Save predictions to: {target_file}')
    with open(f"{target_file}", 'w') as f:
      f.write(json.dumps(pred_list, indent=2))
    return target_file


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-t', '--task', type=str, default='decode_test', help="decode_test/decode_train")
    p.add_argument('-cp', '--ckpt_path', type=str, default=None, help="Model checkpoint")
    p.add_argument('-time', '--timestamp', type=str, default='1', help="Model checkpoint")
    p.add_argument('-f', '--flag', type=str, default='base', help="Model checkpoint")
    p.add_argument('-d', '--dataset', type=str, default="FewShotWoz", 
                   help="specify the dataset: FewShotWoz, SGD")
    p.add_argument('-d2', '--dataset2', type=str, default="FewShotWoz", help="specify the dataset: multiwoz_amr")
    p.add_argument('-m', '--domain', type=str, default="restaurant", 
                   help="specify the domain: restaurant, tv, taxi, ect.")
    p.add_argument('--model_name', type=str, default="scgpt", 
                   help="specify the model name: gpt2-medium, microsoft/DialoGPT-medium")
    p.add_argument('-bz', '--batch_size', type=int, default=4)
    p.add_argument('--train_set', type=str, default="train")
    p.add_argument('--encoder_max_length', type=int, default=80)
    p.add_argument('--decoder_max_length', type=int, default=80)
    p.add_argument('--max_generation_length', type=int, default=80)
    p.add_argument('--num_sequences', type=int, default=5)
    p.add_argument('--top_k', type=int, default=0)
    p.add_argument('--top_p', type=float, default=0.95)
    p.add_argument('--do_sample', type=bool, default=True, 
                   help="specify whether sampling from output distribution during generation")
    p.add_argument('--root_models', type=str, default="models", 
                   help="specify root dir for all model ckpts")
    args = p.parse_args()
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = f"{args.root_models}/{args.model_name}_{args.flag}_{args.dataset}_{args.domain}_{args.train_set}_{args.timestamp}"
    args.ckpt_path = glob.glob(f'{output_dir}/*/')[0] # get checkpoint
    print('Load checkpoint from:', args.ckpt_path)
    decode(args)
