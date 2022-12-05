# -*- coding: utf-8 -*-
# This code comes from: https://github.com/XinnuoXu/SC-GPT/blob/40811cbc060916ef616548f0c283c4b84768de3c/evaluator.py
import re
import json
import operator
import random
import argparse
import numpy as np

from utils.loader.GentScorer import GentScorer
from nltk.tokenize import word_tokenize

random_seed = 1
np.random.seed(random_seed)
random.seed(random_seed)
np.set_printoptions(precision=4)



def sgd_evaluator(args):
    train       = f'../data/{args.dataset}/{args.domain}/train.json'
    valid       = f'../data/{args.dataset}/{args.domain}/dev.json'
    if args.task == 'decode_test':
        test    = f'../data/{args.dataset}/{args.domain}/test.json'
    else:
        test    = f'../data/{args.dataset}/{args.domain}/dev.json'
    vocab       = 'utils/resource/vocab'
    topk        = 1
    print(train, valid, test)
    detectpairs = 'utils/resource/detect.pair'
    target_file = args.target_file
    print(target_file)

    # load train/dev/test data
    gentscorer = GentScorer(detectpairs)
    results_from_gpt = json.load(open(target_file))
    gold_json = json.load(open(test))
    parallel_corpus, log_errors, dacts = [], [], []
    for idx in range(len(results_from_gpt)):
        dact = gold_json[idx][0]
        gold_strs = [gold_json[idx][1]]
        gen_strs = results_from_gpt[idx]
        gen_strs_single = []
        gen_strs_ = []
        for gen_str in gen_strs:
            cl_idx = gen_str.find('<|endoftext|>')
            gen_str = gen_str[:cl_idx].strip().lower()
            gen_str = ' '.join(word_tokenize(gen_str))
            gen_str.replace('-s','')
            gen_str = gen_str.replace('watts','watt -s').replace('televisions','television -s').replace('ports', 'port -s').replace('includes', 'include -s').replace('restaurants','restaurant -s').replace('kids','kid -s').replace('childs','child -s').replace('prices','price -s').replace('range','range -s').\
                replace('laptops','laptop -s').replace('familys','family -s').replace('specifications','specification -s').replace('ratings','rating -s').replace('products','product -s').\
                    replace('constraints','constraint -s').replace('drives','drive -s').replace('dimensions','dimension -s')
            gen_strs_single.append(gen_str)
            gen_strs_.append(gen_str)                    
        
        parallel_corpus.append([gen_strs_[:1], gold_strs])
        dacts.append(dact)
    
    bleuModel, logs = gentscorer.scoreSBLEU(parallel_corpus)
    print(f'FIELNAME: {target_file}, BLEU: {bleuModel}')

    if args.task == 'decode_test':
        output_path = target_file.replace('test_preds.json', 'test_score.json')
    else:
        output_path = target_file.replace('dev_preds.json', 'dev_score.json')
    result = {'BLEU': 100*bleuModel}
    with open(output_path, 'w') as f:
        f.write(json.dumps(result, indent=2))
    return bleuModel



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='decode_test', 
                        help="decode_test/decode_train")
    parser.add_argument('--dataset', type=str, default="SGD", 
                        help="specify the dataset: FewShotWoz, SGD")
    parser.add_argument("--domain", type=str, default="train",
                        help="Please specify a domain")
    parser.add_argument('--timestamp', type=int, default=1, 
                   help="model checkpoint")
    parser.add_argument('-f', '--flag', type=str, default='base', 
                        help="model checkpoint")
    parser.add_argument('-o', "--target_file", default=None, type=str,
                        help="Please specify the result file")
    parser.add_argument('--root_models', type=str, default="models", 
                        help="specify root dir for all model ckpts")
    parser.add_argument('--root_data', type=str, default="data", 
                        help="specify root dir for all generated data")
    args = parser.parse_args()

    output_dir = f"/p/fewshot/{args.root_models}/scgpt_{args.flag}_{args.dataset}_{args.domain}_train_{args.timestamp}"
    if args.task == 'decode_dev':
        args.target_file = f"{output_dir}/dev_preds.json"
    else:
        args.target_file = f"{output_dir}/test_preds.json"
    
    bleu = sgd_evaluator(args)
