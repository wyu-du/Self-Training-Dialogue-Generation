# -*- coding: utf-8 -*-
# This code comes from: https://github.com/XinnuoXu/SC-GPT/blob/40811cbc060916ef616548f0c283c4b84768de3c/evaluator.py
import re
import json
import operator
import random
import argparse
from math import sqrt
import numpy as np

from utils.loader.DataReader import DataReader
from utils.loader.GentScorer import GentScorer
from utils.nlp.nlp import normalize
from nltk.tokenize import word_tokenize

random_seed = 1
np.random.seed(random_seed)
random.seed(random_seed)
np.set_printoptions(precision=4)



def woz_checker(args):
    train       = f'../data/{args.dataset}/{args.domain}/train.json'
    valid       = f'../data/{args.dataset}/{args.domain}/dev.json'
    if args.task == 'decode_test':
        test    = f'../data/{args.dataset}/{args.domain}/test.json'
    else:
        test    = f'../data/{args.dataset}/{args.domain}/dev.json'
    vocab       = 'utils/resource/vocab'
    topk        = 1
    detectpairs = 'utils/resource/detect.pair'
    aug_data_path = f"{args.root_data}/augmented/{args.model_name}_{args.flag}_{args.dataset}_{args.domain}_{args.train_set}_{args.timestamp}"
    with open(f"{aug_data_path}/sa2_augmented_data.json", 'r') as f:
        results_from_gpt = json.load(f)

    # load train/dev/test data
    reader = DataReader(random_seed, args.domain, 'dt', vocab, train, valid, test, 100 , 0, lexCutoff=4)
    gentscorer = GentScorer(detectpairs)

    valid_data = []
    for idx in range(len(results_from_gpt)):
        # read data point
        gen_str = results_from_gpt[idx]['text']['text'].strip().lower()
        gen_str = ' '.join(word_tokenize(gen_str))

        a,sv,s,v,delexed = reader.check(dact=results_from_gpt[idx]['amr'], text=gen_str)
        if -1 in sv or -1 in a: continue
        felements = [reader.cardinality[x+reader.dfs[1]] for x in sv] # [sv_id_1, sv_id_2]
        cnt, total, caty = gentscorer.scoreERR(a,felements, delexed)
        if caty == 0: 
            line = results_from_gpt[idx]['amr'] + ' & ' + results_from_gpt[idx]['text']['text']
            valid_data.append(line)
    return valid_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='decode_test', help="decode_test/decode_train")
    parser.add_argument('--model_name', type=str, default="scgpt", 
                        help="specify the model name: gpt2-medium, microsoft/DialoGPT-medium")
    parser.add_argument('--dataset', type=str, default="FewShotWoz", 
                        help="specify the dataset: FewShotWoz, FewShotSGD")
    parser.add_argument('--train_set', type=str, default="train")
    parser.add_argument('--model_name', type=str, default="scgpt", 
                        help="specify the model name: gpt2-medium, microsoft/DialoGPT-medium")
    parser.add_argument("--domain", type=str, default="taxi", help="Please specify a domain")
    parser.add_argument('--timestamp', type=int, default=2, help="model checkpoint")
    parser.add_argument('-f', '--flag', type=str, default='sa2', help="model checkpoint")
    parser.add_argument('--root_models', type=str, default="models", 
                        help="specify root dir for all model ckpts")
    parser.add_argument('--root_data', type=str, default="data", 
                        help="specify root dir for all generated data")
    args = parser.parse_args()

    valid_data = woz_checker(args)
    print(valid_data)