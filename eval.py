#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.9.19

from metric.metric import * 
import argparse
import gensim
import pickle
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument('--model', type=str, default='HRED', help='model name')
    parser.add_argument('--file', type=str, default=None, help='result file')
    args = parser.parse_args()
    print('eval file:', args.file)

    with open(args.file) as f:
        ref, tgt = [], []
        for idx, line in enumerate(f.readlines()):
            # line = line.lower()
            if idx % 4 == 1:
                line = line.replace("user1", "").replace("user0", "").replace("- ref: ", "").replace('<sos>', '').replace('<eos>', '').strip()
                ref.append(line.split())
            elif idx % 4 == 2:
                line = line.replace("user1", "").replace("user0", "").replace("- tgt: ", "").replace('<sos>', '').replace('<eos>', '').strip()
                tgt.append(line.split())

    assert len(ref) == len(tgt)

    # BLEU and ROUGE
    bleu1_sum, bleu2_sum, bleu3_sum, bleu4_sum = 0, 0, 0, 0 
        
    refs, tgts = [' '.join(i) for i in ref], [' '.join(i) for i in tgt]
    bleu1_sum, bleu2_sum, bleu3_sum, bleu4_sum = cal_BLEU(refs, tgts) 

    # Distinct-1, Distinct-2
    candidates, references = [], []
    for line1, line2 in zip(tgt, ref):
        candidates.extend(line1)
        references.extend(line2)
    distinct_1, distinct_2 = cal_Distinct(candidates)
    rdistinct_1, rdistinct_2 = cal_Distinct(references)

    intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(tgt)
    #rintra_dist1, rintra_dist2, rinter_dist1, rinter_dist2 = distinct(ref)

    # BERTScore < 512 for bert
    # Fuck BERTScore, slow as the snail, fuck it
    bert_scores = cal_BERTScore(refs, tgts)
    
    # Embedding-based metric: Embedding Average (EA), Vector Extrema (VX), Greedy Matching (GM)
    # load the dict
    dic = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
    print('[!] load the GoogleNews 300 word2vector by gensim over')
     
    
    keys = dic.vocab
    samples = [sent for sent in tgt]
    ground_truth = [sent for sent in ref]
    samples = [[dic[s] for s in sent if s in keys] for sent in samples]
    ground_truth = [[dic[s] for s in sent if s in keys] for sent in ground_truth]
    indices = [i for i, s, g in zip(range(len(samples)), samples, ground_truth) if s != [] and g != []]
    samples = [samples[i] for i in indices]
    ground_truth = [ground_truth[i] for i in indices]
    n_sent = len(samples)


    metric_average = embedding_metric(samples, ground_truth, dic, 'average')
    metric_extrema = embedding_metric(samples, ground_truth, dic, 'extrema')
    metric_greedy = embedding_metric(samples, ground_truth, dic, 'greedy')
    length_tgt = []
    for sent in tgt:
        length_tgt.append(len(sent))
    length_ref = []
    for sent in ref:
        length_ref.append(len(sent)) 
    times = 100
    print(f'Model {args.model} Result')
    print(f'BLEU-1: {round(bleu1_sum, 4)*times}%')
    print(f'BLEU-2: {round(bleu2_sum, 4)*times}%')
    print(f'BLEU-3: {round(bleu3_sum, 4)*times}%')
    print(f'BLEU-4: {round(bleu4_sum, 4)*times}%') 
    print(f'Distinct-1: {round(distinct_1, 4)*times}%; Distinct-2: {round(distinct_2, 4)*times}%')
    print(f'Ref distinct-1: {round(rdistinct_1, 4)*times}%; Ref distinct-2: {round(rdistinct_2, 4)*times}%')  
    
    print(f'Metric_Average: {round(np.mean(metric_average), 4)*times}%')
    print(f'Metric_Extrema: {round(np.mean(metric_extrema), 4)*times}%')
    print(f'Metric_Greedy: {round(np.mean(metric_greedy), 4)*times}%')


    print(f'--------------------------------------------------------------') 
    print(f'Intra-Distinct-1: {round(intra_dist1, 4)*times}%')
    print(f'Intra-Distinct-2: {round(intra_dist2, 4)*times}%')
    print(f'Inter-Distinct-1: {round(inter_dist1, 4)*times}%')
    print(f'Inter-Distinct-2: {round(inter_dist2, 4)*times}%')

    #print(f'Ref-Intra-Distinct-1: {round(rintra_dist1, 4)}')
    #print(f'Ref-Intra-Distinct-2: {round(rintra_dist2, 4)}')
    #print(f'Ref-Inter-Distinct-1: {round(rinter_dist1, 4)}')
    #print(f'Ref-Inter-Distinct-2: {round(rinter_dist2, 4)}')
 
    print(f'Length-Average-tgt: {round(np.mean(length_tgt), 4)}')
    print(f'Length-Average-ref: {round(np.mean(length_ref), 4)}') 




    





