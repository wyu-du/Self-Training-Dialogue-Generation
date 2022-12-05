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



def woz_evaluator(args):
    train       = f'../data/{args.dataset}/{args.domain}/train.json'
    valid       = f'../data/{args.dataset}/{args.domain}/dev.json'
    if args.task == 'decode_test':
        test    = f'../data/{args.dataset}/{args.domain}/test.json'
    else:
        test    = f'../data/{args.dataset}/{args.domain}/dev.json'
    vocab       = 'utils/resource/vocab'
    topk        = 1
    # print(train, valid, test)
    detectpairs = 'utils/resource/detect.pair'
    target_file = args.target_file
    # print(target_file)
    # load train/dev/test data
    reader = DataReader(random_seed, args.domain, 'dt', vocab, train, valid, test, 100 , 0, lexCutoff=4)
    gentscorer = GentScorer(detectpairs)

    results_from_gpt = json.load(open(target_file))
    idx = 0
    parallel_corpus, log_errors, dacts = [], [], []
    gencnts, refcnts = [0.0,0.0,0.0],[0.0,0.0,0.0]
    counts = 0
    while True:
        # read data point
        data = reader.read(mode='test',batch=1)
        if data==None: break
        a,sv,s,v,sents,dact,bases,_,_ = data
        
        # remove batch dimension
        a,sv,s,v = a[0],sv[0],s[0],v[0]
        sents,dact,bases = sents[0],dact[0],bases[0]
        
        gen_strs = results_from_gpt[idx]
        # gen_strs_single = []
        gen_strs_ = []
        for gen_str in gen_strs:
            cl_idx = gen_str.find('<|endoftext|>')
            gen_str = gen_str[:cl_idx].strip().lower()
            gen_str = ' '.join(word_tokenize(gen_str))
            # gen_str.replace('-s','')
            # gen_str = gen_str.replace('watts','watt -s').replace('televisions','television -s').replace('ports', 'port -s').replace('includes', 'include -s').replace('restaurants','restaurant -s').replace('kids','kid -s').replace('childs','child -s').replace('prices','price -s').replace('range','range -s').\
            #     replace('laptops','laptop -s').replace('familys','family -s').replace('specifications','specification -s').replace('ratings','rating -s').replace('products','product -s').\
            #         replace('constraints','constraint -s').replace('drives','drive -s').replace('dimensions','dimension -s')
            # gen_strs_single.append(gen_str)
            gen_strs_.append(gen_str)                    
            
        gens = gen_strs_
        idx += 1
        felements = [reader.cardinality[x+reader.dfs[1]] for x in sv] # [sv_id_1, sv_id_2]
        gens_with_penalty = []

        for i in range(len(gens)):
            # score slot error rate
            delexed = reader.delexicalise(normalize(re.sub(' [\.\?\!]$','',gens[i])), dact)
            cnt, total, caty = gentscorer.scoreERR(a,felements, delexed)
            gens_with_penalty.append((total, len(gens[i].split()), gens[i]))
        gens_with_penalty = sorted(gens_with_penalty, key=lambda k:k[0])[:topk]

        # pick the top-1 sentence with lowest categorical error
        gens = [g[2] for g in gens_with_penalty][:1]

        total_ex = 0; cnt_ex = 0
        for i in range(len(gens)):
            # score slot error rate
            delexed = reader.delexicalise(normalize(re.sub(' [\.\?\!]$','',gens[i])), dact)
            cnt, total, caty = gentscorer.scoreERR(a,felements, delexed)
            gens[i] = reader.lexicalise(delexed,dact)
            # accumulate slot error cnts
            gencnts[0]  += cnt
            gencnts[1]  += total
            gencnts[2]  += caty
            total_ex += total
            cnt_ex += cnt
        if cnt_ex == 0:
            #log_errors.append(-1)
            log_errors.append([-1, total_ex, cnt_ex])
        else:
            # log_errors.append(total_ex/cnt_ex)
            log_errors.append([total_ex/cnt_ex, total_ex, cnt_ex, felements, delexed, sents[0]])
        
        # compute gold standard slot error rate
        for sent in sents:
            # score slot error rate
            cnt, total, caty = gentscorer.scoreERR(a,felements,
                    reader.delexicalise(normalize(re.sub(' [\.\?\!]$','',sent)),dact))
            # accumulate slot error cnts
            refcnts[0]  += cnt    # total slot count
            refcnts[1]  += total  # caty_slot_error + bnay_slot_error
            refcnts[2]  += caty   # caty_slot_error
            if caty > 0:
                counts+=1

        parallel_corpus.append([[g for g in gens], sents])
        dacts.append(dact)
    
    predicted_sentences = []
    for i in parallel_corpus:
        predicted_sentences.append(i[0][0])

    bleuModel, logs = gentscorer.scoreSBLEU(parallel_corpus)

    # print ('##############################################')
    # print ('BLEU SCORE & SLOT ERROR on GENERATED SENTENCES')
    # print ('##############################################')
    # print ('Metric       :\tBLEU\tT.ERR\tA.ERR')
    # print ('Ref          :\t%.1f\t%2.2f%%\t%2.2f%%'% (100.0, 100*refcnts[1]/refcnts[0],100*refcnts[2]/refcnts[0]))
    # print ('----------------------------------------------')
    # print ('This Model   :\t%.4f\t%2.2f%%\t%2.2f%%'% (100*bleuModel, 100*gencnts[1]/gencnts[0],100*gencnts[2]/gencnts[0]))
    # print(f'FIELNAME: {target_file}, BLEU: {100*bleuModel}, ERR:{100*gencnts[1]/gencnts[0]}')

    if args.task == 'decode_test':
        output_path = target_file.replace('test_preds.json', 'test_score.json')
    else:
        output_path = target_file.replace('dev_preds.json', 'dev_score.json')
    result = {'BLEU': 100*bleuModel, 'ERR': 100*gencnts[1]/gencnts[0]}
    with open(output_path, 'w') as f:
        f.write(json.dumps(result, indent=2))
    return bleuModel, 100*gencnts[1]/gencnts[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='decode_test', help="decode_test/decode_dev")
    parser.add_argument('--model_name', type=str, default="scgpt", 
                        help="specify the model name: gpt2-medium, microsoft/DialoGPT-medium")
    parser.add_argument('--dataset', type=str, default="FewShotWoz", 
                        help="specify the dataset: FewShotWoz, FewShotSGD")
    parser.add_argument('-m', '--domain', type=str, default="restaurant", 
                        help="specify the domain: restaurant, tv, taxi, ect.")
    parser.add_argument('--timestamp', type=int, default=1, help="model checkpoint")
    parser.add_argument('-f', '--flag', type=str, default='base', help="model checkpoint")
    parser.add_argument('-o', "--target_file", default=None, type=str, help="Please specify the result file")
    parser.add_argument('--train_set', type=str, default="train")
    parser.add_argument('--root_models', type=str, default="models", help="specify root dir for all model ckpts")
    args = parser.parse_args()

    output_dir = f"{args.root_models}/{args.model_name}_{args.flag}_{args.dataset}_{args.domain}_{args.train_set}_{args.timestamp}"
    if args.task == 'decode_dev':
        args.target_file = f"{output_dir}/dev_preds.json"
    else:
        args.target_file = f"{output_dir}/test_preds.json"
    
    bleu, err = woz_evaluator(args)
