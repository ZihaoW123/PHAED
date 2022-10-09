#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.9.14

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import transformers
from utils import *
import ipdb
import random
import nltk
import os
import pickle
    

def load_data_flatten(src, tgt, src_vocab, tgt_vocab, maxlen, tgt_maxlen):
    '''
    Used by vanilla seq2seq with attention and transformer
    '''
    # check the file, exist -> ignore
    src_prepath = os.path.splitext(src)[0] + '-flatten.pkl'
    tgt_prepath = os.path.splitext(tgt)[0] + '-flatten.pkl'
    if os.path.exists(src_prepath) and os.path.exists(tgt_prepath):
        print(f'[!] preprocessed file {src_prepath} exist, load directly')
        print(f'[!] preprocessed file {tgt_prepath} exist, load directly')
        with open(src_prepath, 'rb') as f:
            src_dataset = pickle.load(f)
        with open(tgt_prepath, 'rb') as f:
            tgt_dataset = pickle.load(f)
        return src_dataset, tgt_dataset
    else:
        print(f'[!] cannot find the preprocessed file')
    
    # sort by the lengths
    src_w2idx, src_idx2w = load_pickle(src_vocab)
    tgt_w2idx, tgt_idx2w = load_pickle(tgt_vocab)

    # sub function
    def load_(filename, w2idx, src=True):
        with open(filename) as f:
            dataset = []
            for line in tqdm(f.readlines()):
                line = clean(line)
                # if '<user0>' in line: user_c = '<user0>'
                # elif '<user1>' in line: user_c = '<user1>'
                line = line.replace('<user0>', 'user0')
                line = line.replace('<user1>', 'user1')
                line = [w2idx['<sos>']] + [w2idx.get(w, w2idx['<unk>']) for w in nltk.word_tokenize(line)] + [w2idx['<eos>']]
                if src and len(line) > maxlen:
                    line = [w2idx['<sos>']] + line[-maxlen:]
                elif src == False and len(line) > tgt_maxlen:
                    line = line[:tgt_maxlen] + [w2idx['<eos>']]
                dataset.append(line)
        return dataset

    src_dataset = load_(src, src_w2idx, src=True)    # [datasize, lengths]
    tgt_dataset = load_(tgt, tgt_w2idx, src=False)    # [datasize, lengths]
    print(f'[!] load dataset over, write into file {src_prepath} and {tgt_prepath}')
    
    with open(src_prepath, 'wb') as f:
        pickle.dump(src_dataset, f)
    with open(tgt_prepath, 'wb') as f:
        pickle.dump(tgt_dataset, f)

    return src_dataset, tgt_dataset


def get_batch_data(src, tgt, src_vocab, tgt_vocab, batch_size, maxlen, tgt_maxlen,
                   plus=0, ld=True):
    # batch and convert to tensor for training
    # batch according to the turns
    # [datasize, turns, lengths], [datasize, lengths]
    src_w2idx, src_idx2w = load_pickle(src_vocab)
    tgt_w2idx, tgt_idx2w = load_pickle(tgt_vocab)
    
    src_dataset, _, tgt_dataset, _ = load_data(src, tgt, src_vocab, tgt_vocab, maxlen, tgt_maxlen, ld=ld)
    turns = [len(dialog) for dialog in src_dataset]
    turnidx = np.argsort(turns)
    # sort by the lengrh of the turns
    src_dataset = [src_dataset[idx] for idx in turnidx]
    tgt_dataset = [tgt_dataset[idx] for idx in turnidx]
    # print(f'[!] dataset size: {len(src_dataset)}')

    # batch and convert to tensor
    turns = [len(dialog) for dialog in src_dataset]
    fidx, bidx = 0, 0
    while fidx < len(src_dataset):
        bidx = fidx + batch_size
        head = turns[fidx]
        cidx = 10000
        for p, i in enumerate(turns[fidx:bidx]):
            if i != head:
                cidx = p
                break
        cidx = fidx + cidx
        bidx = min(bidx, cidx)
        # print(fidx, bidx)

        # batch, [batch, turns, lengths], [batch, lengths]
        # shuffle
        sbatch, tbatch = src_dataset[fidx:bidx], tgt_dataset[fidx:bidx]
        
        if len(sbatch[0]) <= plus:
            fidx = bidx
            continue
        
        shuffleidx = np.arange(0, len(sbatch))
        np.random.shuffle(shuffleidx)
        sbatch = [sbatch[idx] for idx in shuffleidx]
        tbatch = [tbatch[idx] for idx in shuffleidx]

        # convert to [turns, batch, lengths], [batch, lengths]
        sbatch = transformer_list(sbatch)
        bs, ts = len(sbatch[0]), len(sbatch)

        # pad src by turns
        # create the lengths: [turns, batch] for sbatch
        turn_lengths = []
        for i in range(ts):
            lengths = []
            for item in sbatch[i]:
                lengths.append(len(item))
            turn_lengths.append(lengths)
            pad_sequence(src_w2idx['<pad>'], sbatch[i], bs)

        # pad tgt, [batch, turns, lengths]
        pad_sequence(tgt_w2idx['<pad>'], tbatch, bs)
        
        # convert to tensor, change to cuda version tensor
        srcbatch = []
        for i in range(ts):
            pause = torch.tensor(sbatch[i], dtype=torch.long).transpose(0, 1)
            if torch.cuda.is_available():
                pause = pause.cuda()
            srcbatch.append(pause)
        sbatch = srcbatch
        tbatch = torch.tensor(tbatch, dtype=torch.long).transpose(0, 1)
        if torch.cuda.is_available():
            tbatch = tbatch.cuda()

        turn_lengths = torch.tensor(turn_lengths, dtype=torch.long)
        if torch.cuda.is_available():
            turn_lengths = turn_lengths.cuda()

        fidx = bidx

        yield sbatch, tbatch, turn_lengths


def get_batch_data_flatten(src, tgt, src_vocab, tgt_vocab, batch_size, maxlen, tgt_maxlen):
    # flatten batch data for unHRED-based models (Seq2Seq)
    # return long context for predicting response
    # [datasize, turns, lengths], [datasize, lengths]
    src_w2idx, src_idx2w = load_pickle(src_vocab)
    tgt_w2idx, tgt_idx2w = load_pickle(tgt_vocab)
    
    # [datasize, lengths], [datasize, lengths]
    src_dataset, tgt_dataset = load_data_flatten(src, tgt, src_vocab, tgt_vocab, maxlen, tgt_maxlen)

    turns = [len(i) for i in src_dataset]
    turnsidx = np.argsort(turns)

    # sort by the lengths
    src_dataset = [src_dataset[i] for i in turnsidx]
    tgt_dataset = [tgt_dataset[i] for i in turnsidx]

    # generate the batch
    turns = [len(i) for i in src_dataset]
    fidx, bidx = 0, 0
    while fidx < len(src_dataset):
        bidx = fidx + batch_size
        sbatch, tbatch = src_dataset[fidx:bidx], tgt_dataset[fidx:bidx]
        # shuffle
        shuffleidx = np.arange(0, len(sbatch))
        np.random.shuffle(shuffleidx)
        sbatch = [sbatch[idx] for idx in shuffleidx]
        tbatch = [tbatch[idx] for idx in shuffleidx]
        
        bs = len(sbatch)

        # pad sbatch and tbatch
        turn_lengths = [len(sbatch[i]) for i in range(bs)]
        pad_sequence(src_w2idx['<pad>'], sbatch, bs)
        pad_sequence(tgt_w2idx['<pad>'], tbatch, bs)
        
        # [seq_len, batch]
        sbatch = torch.tensor(sbatch, dtype=torch.long).transpose(0, 1)
        tbatch = torch.tensor(tbatch, dtype=torch.long).transpose(0, 1)
        turn_lengths = torch.tensor(turn_lengths, dtype=torch.long)
        if torch.cuda.is_available():
            tbatch = tbatch.cuda()
            sbatch = sbatch.cuda()
            turn_lengths = turn_lengths.cuda()

        fidx = bidx

        yield sbatch, tbatch, turn_lengths
        
        
# transformers
def load_data_flatten_tf(src, tgt, maxlen, tokenizer):
    with open(src) as f:
        src_corpus = []
        for line in tqdm(f.readlines()):
            line = clean(line)
            line = line.replace('<user0>', '').replace('<user1>', '')
            line = [tokenizer.cls_token_id] + [tokenizer.convert_tokens_to_ids(w) for w in nltk.word_tokenize(line)] + [tokenizer.sep_token_id]
            if len(line) > maxlen:
                line = [line[0]] + line[-maxlen:]
            src_corpus.append(line)
    with open(tgt) as f:
        tgt_corpus = []
        for line in tqdm(f.readlines()):
            line = clean(line)
            line = [tokenizer.cls_token_id] + [tokenizer.convert_tokens_to_ids(w) for w in nltk.word_tokenize(line)] + [tokenizer.sep_token_id]
            if len(line) > maxlen:
                line = line[:maxlen]
            tgt_corpus.append(line)
    return src_corpus, tgt_corpus
    
        
def get_batch_data_flatten_tf(src, tgt, batch_size, maxlen):
    '''
    1. No turn_lengths because of transformer
    2. src_key_padding_mask and tgt_key_padding_mask
    '''
    # for transformer, return
    # src_mask and tgt_mask
    tokenizer = transformers.BertTokenizer.from_pretrained('config/vocab_en.txt')
    
    # [datasize, lengths], [datasize, lengths]
    src_dataset, tgt_dataset = load_data_flatten_tf(src, tgt, maxlen, tokenizer)

    turns = [len(i) for i in src_dataset]
    turnsidx = np.argsort(turns)

    # sort by the lengths
    src_dataset = [src_dataset[i] for i in turnsidx]
    tgt_dataset = [tgt_dataset[i] for i in turnsidx]

    # generate the batch
    turns = [len(i) for i in src_dataset]
    fidx, bidx = 0, 0
    while fidx < len(src_dataset):
        bidx = fidx + batch_size
        sbatch, tbatch = src_dataset[fidx:bidx], tgt_dataset[fidx:bidx]
        # shuffle
        shuffleidx = np.arange(0, len(sbatch))
        np.random.shuffle(shuffleidx)
        sbatch = [sbatch[idx] for idx in shuffleidx]
        tbatch = [tbatch[idx] for idx in shuffleidx]
        
        bs = len(sbatch)

        # pad sbatch and tbatch, [batch, seq]
        pad_sequence(tokenizer.pad_token_id, sbatch, bs)
        pad_sequence(tokenizer.pad_token_id, tbatch, bs)
        
        # [batch, seq]
        sbatch = torch.tensor(sbatch, dtype=torch.long)
        tbatch = torch.tensor(tbatch, dtype=torch.long)
        if torch.cuda.is_available():
            tbatch = tbatch.cuda()
            sbatch = sbatch.cuda()

        fidx = bidx

        yield sbatch, tbatch
        

def get_batch_data_graph(src, tgt, graph, src_vocab, tgt_vocab, 
                         batch_size, maxlen, tgt_maxlen, plus=0):
    '''get batch data of hierarchical and graph mode
    return data:
    - sbatch: [turn, batch, length]
    - tbatch: [batch, length]
    - gbatch: [batch, ([2, num_edge], [num_edge])]
    - turn_lengths: [batch]
    '''
    src_w2idx, src_idx2w = load_pickle(src_vocab)
    tgt_w2idx, tgt_idx2w = load_pickle(tgt_vocab)
    
    src_dataset, src_user, tgt_dataset, tgt_user = load_data(src, tgt, src_vocab, tgt_vocab, maxlen, tgt_maxlen)
    graph = load_pickle(graph)
    
    turns = [len(dialog) for dialog in src_dataset]
    turnidx = np.argsort(turns)
    
    # sort by the lengrh of the turns
    src_dataset = [src_dataset[idx] for idx in turnidx]
    tgt_dataset = [tgt_dataset[idx] for idx in turnidx]
    graph = [graph[idx] for idx in turnidx]
    src_user = [src_user[idx] for idx in turnidx]
    tgt_user = [tgt_user[idx] for idx in turnidx]
    # print(f'[!] dataset size: {len(src_dataset)}')

    # batch and convert to tensor
    turns = [len(dialog) for dialog in src_dataset]
    fidx, bidx = 0, 0
    while fidx < len(src_dataset):
        bidx = fidx + batch_size
        head = turns[fidx]
        cidx = 10000
        for p, i in enumerate(turns[fidx:bidx]):
            if i != head:
                cidx = p
                break
        cidx = fidx + cidx
        bidx = min(bidx, cidx)
        
        sbatch, tbatch, gbatch = src_dataset[fidx:bidx], tgt_dataset[fidx:bidx], graph[fidx:bidx]
        subatch, tubatch = src_user[fidx:bidx], tgt_user[fidx:bidx]
        
        if len(sbatch[0]) <= plus:
            fidx = bidx
            continue
            
        shuffleidx = np.arange(0, len(sbatch))
        np.random.shuffle(shuffleidx)
        sbatch = [sbatch[idx] for idx in shuffleidx]   # [batch, turns, lengths]
        tbatch = [tbatch[idx] for idx in shuffleidx]   # [batch, lengths]
        gbatch = [gbatch[idx] for idx in shuffleidx]   # [batch, ([2, edges_num], [edges_num]),]
        
        sbatch = transformer_list(sbatch)    # [turns, batch, lengths]
        bs, ts = len(sbatch[0]), len(sbatch)
        
        turn_lengths = []
        for i in range(ts):
            lengths = []
            for item in sbatch[i]:
                lengths.append(len(item))
            turn_lengths.append(lengths)
            pad_sequence(src_w2idx['<pad>'], sbatch[i], bs)

        pad_sequence(tgt_w2idx['<pad>'], tbatch, bs)
        
        # convert to tensor
        srcbatch = []
        for i in range(ts):
            pause = torch.tensor(sbatch[i], dtype=torch.long).transpose(0, 1)
            if torch.cuda.is_available():
                pause = pause.cuda()
            srcbatch.append(pause)    # [turns, seq_len, batch]
        sbatch = srcbatch
        tbatch = torch.tensor(tbatch, dtype=torch.long).transpose(0, 1)    # [seq_len, batch]
        subatch = torch.tensor(subatch, dtype=torch.long).transpose(0, 1)   # [turns, batch]
        tubatch = torch.tensor(tubatch, dtype=torch.long)    # [batch]
        
        turn_lengths = torch.tensor(turn_lengths, dtype=torch.long)     # [batch]
        if torch.cuda.is_available():
            tbatch = tbatch.cuda()
            turn_lengths = turn_lengths.cuda()
            subatch = subatch.cuda()
            tubatch = tubatch.cuda()
            
        fidx = bidx
        yield sbatch, tbatch, gbatch, subatch, tubatch, turn_lengths


# hi-transformers
from copy import deepcopy
def src_doc_sents_map(src_seqs: torch.Tensor):
    sent_mapping = src_seqs.tolist() ## batch, length
    sent_rank = deepcopy(sent_mapping)
    max_n_sents = 0
    for i in range(len(sent_mapping)):
        n_sents = -1
        cur_rank = 0
        for j in range(len(sent_mapping[i])):
            id = sent_mapping[i][j]
            if id == SOS_ID:
                n_sents += 1
                cur_rank = 0
            sent_mapping[i][j] = n_sents
            sent_rank[i][j] = cur_rank
            if id !=  PAD_ID:
                cur_rank += 1
            else:
                sent_rank[i][j] = PAD_ID
        if max_n_sents < n_sents:
            max_n_sents = n_sents
    res = torch.tensor(sent_mapping).detach()
    sent_rank = torch.tensor(sent_rank, dtype=torch.float32, device=res.device)
    if torch.cuda.is_available():
        res = res.cuda()
        sent_rank = sent_rank.cuda()
    return res, sent_rank, max_n_sents

def get_batch_data_hier_tf(src, tgt, src_vocab, tgt_vocab, batch_size, maxlen, tgt_maxlen,
                   plus=0, ld=True, shuffle=True, random=True):
    # batch and convert to tensor for training
    # batch according to the turns
    # [datasize, turns, lengths], [datasize, lengths]
    src_w2idx, src_idx2w = load_pickle(src_vocab)
    tgt_w2idx, tgt_idx2w = load_pickle(tgt_vocab)

    src_dataset, tgt_dataset, src_user, tgt_user, src_turn_id, tgt_turn_id = load_data_users(src, tgt, src_vocab, tgt_vocab, maxlen, tgt_maxlen, ld=ld)
    # src_dataset, tgt_dataset, src_user, tgt_user, src_turn_id, tgt_turn_id = load_data_users_wo_userid(src, tgt, src_vocab, tgt_vocab, maxlen, tgt_maxlen, ld=ld)
    turns = [len(dialog) for dialog in src_dataset]
    turns_tgt = [len(dialog) for dialog in tgt_dataset]
    turnidx = np.argsort(np.array(turns)*1000000 + np.array(turns_tgt))
    if random==False:
        turnidx = np.arange(0, len(turns))
    # sort by the lengrh of the turns
    src_dataset = [src_dataset[idx] for idx in turnidx]
    tgt_dataset = [tgt_dataset[idx] for idx in turnidx]
    src_turn_id = [src_turn_id[idx] for idx in turnidx]
    tgt_turn_id = [tgt_turn_id[idx] for idx in turnidx]
    src_user = [src_user[idx] for idx in turnidx]
    tgt_user = [tgt_user[idx] for idx in turnidx]
    # print(f'[!] dataset size: {len(src_dataset)}')

    # batch and convert to tensor
    turns = [len(dialog) for dialog in src_dataset]
    turns_tgt = [len(dialog) for dialog in tgt_dataset]
    fidx, bidx = 0, 0
    while fidx < len(src_dataset):
        bidx = fidx + batch_size
        head = turns[fidx]
        head_tgt = turns_tgt[fidx]
        cidx = 10000
        i_tgt = turns_tgt[fidx:bidx]
        for p, i in enumerate(turns[fidx:bidx]):
            if i != head or i_tgt[p] != head_tgt:
                cidx = p
                break
        cidx = fidx + cidx
        bidx = min(bidx, cidx)
        # print(fidx, bidx)

        # batch, [batch, turns, lengths], [batch, turns, lengths]
        # shuffle
        sbatch, tbatch,  = src_dataset[fidx:bidx], tgt_dataset[fidx:bidx],
        sbatch_user, tbatch_user,  = src_user[fidx:bidx], tgt_user[fidx:bidx],
        sbatch_turn_id, tbatch_turn_id,  = src_turn_id[fidx:bidx], tgt_turn_id[fidx:bidx],

        if len(sbatch[0]) <= plus:
            fidx = bidx
            continue

        shuffleidx = np.arange(0, len(sbatch))
        if random:
            np.random.shuffle(shuffleidx)
        sbatch = [sbatch[idx] for idx in shuffleidx]
        tbatch = [tbatch[idx] for idx in shuffleidx]
        sbatch_user = [sbatch_user[idx] for idx in shuffleidx]
        tbatch_user = [tbatch_user[idx] for idx in shuffleidx]  ##[batch, turns, length]
        sbatch_turn_id= [sbatch_turn_id[idx] for idx in shuffleidx]
        tbatch_turn_id = [tbatch_turn_id[idx] for idx in shuffleidx]
        assert len(sbatch[0]) <= len(tbatch[0])

        # convert to [turns, batch, lengths], [turns, batch, lengths]

        # pad src by turns
        # create the lengths: [turns, batch] for sbatch
        # sbatch: [turns, batch, lengths]
        pad_src_turn_num = len(tbatch[0]) - len(sbatch[0])
        if pad_src_turn_num > 0:
            sbatch_ = [[src_w2idx['<sos>'], PAD_U_ID, src_w2idx['<eos>']] for i in range(pad_src_turn_num)]
            id_max = max(tbatch_turn_id[0])
            sbatch_turn_id_ = [tbatch_turn_id[0][i] for i in range(pad_src_turn_num)]
            user_id = sbatch_user[0][0]
            sbatch_user_ = [user_id for i in range(pad_src_turn_num)]
            for i in range(len(sbatch)):
                sbatch[i] = sbatch_ + sbatch[i]
                sbatch_turn_id[i] = sbatch_turn_id_ + sbatch_turn_id[i]
                sbatch_user[i] = sbatch_user_ + sbatch_user[i]
                
        sbatch_turn_id= [tmp[::-1] for tmp in sbatch_turn_id]
        tbatch_turn_id = [tmp[::-1] for tmp in tbatch_turn_id]

        src_turn_lengths = []
        for i in range(len(sbatch)):
            lengths = []
            for item in sbatch[i]:
                lengths.append(len(item))
            src_turn_lengths.append(lengths)

        sbatch_ = []
        sbatch_turn_id_ = []
        for i in range(len(sbatch)):
            sent = []
            turn_id = []
            for j in range(len(sbatch[i])):
                sent += sbatch[i][j]
                turn_id += ([sbatch_turn_id[i][j]] * len(sbatch[i][j]))
            sbatch_.append(sent)
            sbatch_turn_id_.append(turn_id)
        sbatch = sbatch_
        sbatch_turn_id = sbatch_turn_id_
        pad_sequence(src_w2idx['<pad>'], sbatch, len(sbatch), maxlen=None)
        pad_sequence(sbatch_turn_id[-1][-1], sbatch_turn_id, len(sbatch_turn_id), maxlen=None)
        assert len(sbatch[0]) == len(sbatch_turn_id[0])

        tgt_turn_lengths = []
        for i in range(len(tbatch)):
            lengths = []
            for item in tbatch[i]:
                lengths.append(len(item))
            tgt_turn_lengths.append(lengths)
        # maxlen_tbatch = np.max(tgt_turn_lengths)
        # for i in range(len(tbatch)):
        #     pad_sequence(tgt_w2idx['<pad>'], tbatch[i], len(tbatch[0]), maxlen=None)

        # convert to tensor, change to cuda version tensor

        # sbatch [batch, turns_src*lengths]
        srcbatch = []
        for i in range(len(sbatch)):
            pause = torch.tensor(sbatch[i], dtype=torch.long)
            if torch.cuda.is_available():
                pause = pause.cuda()
            srcbatch.append(pause)
        sbatch = torch.stack(srcbatch)

        tgtbatch = []
        tbatch = transformer_list(tbatch)
        bs, ts = len(tbatch[0]), len(tbatch)
        # tbatch [batch, turns_tgt, lengths]
        for i in range(len(tbatch)):
            pad_sequence(tgt_w2idx['<pad>'], tbatch[i], len(tbatch[i]), maxlen=None) # tgt_maxlen
            pause = torch.tensor(tbatch[i], dtype=torch.long)
            if torch.cuda.is_available():
                pause = pause.cuda()
            tgtbatch.append(pause)
        tbatch = tgtbatch # torch.stack(tgtbatch)



        src_turn_lengths = torch.tensor(src_turn_lengths, dtype=torch.long)
        tgt_turn_lengths = torch.tensor(tgt_turn_lengths, dtype=torch.long)
        sbatch_user = torch.tensor(sbatch_user, dtype=torch.long) #[batch, turns_src]
        tbatch_user = torch.tensor(tbatch_user, dtype=torch.long) #[batch, turns_tgt]
        sbatch_turn_id = torch.tensor(sbatch_turn_id, dtype=torch.long)
        tbatch_turn_id = torch.tensor(tbatch_turn_id, dtype=torch.long)

        fidx = bidx
        sents_mapping, sent_rank, _ = src_doc_sents_map(sbatch)
        if torch.cuda.is_available():
            src_turn_lengths = src_turn_lengths.cuda()
            tgt_turn_lengths = tgt_turn_lengths.cuda()
            sbatch_turn_id = sbatch_turn_id.cuda()
            tbatch_turn_id = tbatch_turn_id.cuda()
            sents_mapping = sents_mapping.cuda()
            sent_rank = sent_rank.cuda()

        yield (sbatch, sents_mapping, sent_rank), tbatch, \
              (src_turn_lengths, tgt_turn_lengths,
               sbatch_user, tbatch_user,
               sbatch_turn_id, tbatch_turn_id)

def get_batch_data_hier_tf_random(src, tgt, src_vocab, tgt_vocab, batch_size, maxlen, tgt_maxlen,
                           plus=0, ld=True, shuffle=True):
    # batch and convert to tensor for training
    # batch according to the turns
    # [datasize, turns, lengths], [datasize, lengths]
    src_w2idx, src_idx2w = load_pickle(src_vocab)
    tgt_w2idx, tgt_idx2w = load_pickle(tgt_vocab)

    src_dataset, tgt_dataset, src_user, tgt_user, src_turn_id, tgt_turn_id = load_data_users(src, tgt, src_vocab, tgt_vocab, maxlen, tgt_maxlen, ld=ld)
    # src_dataset, tgt_dataset, src_user, tgt_user, src_turn_id, tgt_turn_id = load_data_users_wo_userid(src, tgt, src_vocab, tgt_vocab, maxlen, tgt_maxlen, ld=ld)
    turns = [len(dialog) for dialog in src_dataset]
    turns_tgt = [len(dialog) for dialog in tgt_dataset]
    turnidx = np.argsort(np.array(turns) * 1000000 + np.array(turns_tgt))
    # sort by the lengrh of the turns
    src_dataset = [src_dataset[idx] for idx in turnidx]
    tgt_dataset = [tgt_dataset[idx] for idx in turnidx]
    src_turn_id = [src_turn_id[idx] for idx in turnidx]
    tgt_turn_id = [tgt_turn_id[idx] for idx in turnidx]
    src_user = [src_user[idx] for idx in turnidx]
    tgt_user = [tgt_user[idx] for idx in turnidx]
    # print(f'[!] dataset size: {len(src_dataset)}')

    # batch and convert to tensor
    turns = [len(dialog) for dialog in src_dataset]
    turns_tgt = [len(dialog) for dialog in tgt_dataset]
    fidx, bidx = 0, 0
    fidx_bidx_list = []
    while fidx < len(src_dataset):
        bidx = fidx + batch_size
        head = turns[fidx]
        head_tgt = turns_tgt[fidx]
        cidx = 10000
        i_tgt = turns_tgt[fidx:bidx]
        for p, i in enumerate(turns[fidx:bidx]):
            if i != head or i_tgt[p] != head_tgt:
                cidx = p
                break
        cidx = fidx + cidx
        bidx = min(bidx, cidx)
        fidx_bidx_list.append([fidx, bidx])
        fidx = bidx
        # print(fidx, bidx)

    shuffleidx = np.arange(0, len(fidx_bidx_list))
    np.random.shuffle(shuffleidx)
    fidx_bidx_list_ = [fidx_bidx_list[i] for i in shuffleidx]

    for fidx, bidx in fidx_bidx_list_:
        # batch, [batch, turns, lengths], [batch, turns, lengths]
        # shuffle
        sbatch, tbatch, = src_dataset[fidx:bidx], tgt_dataset[fidx:bidx],
        sbatch_user, tbatch_user, = src_user[fidx:bidx], tgt_user[fidx:bidx],
        sbatch_turn_id, tbatch_turn_id, = src_turn_id[fidx:bidx], tgt_turn_id[fidx:bidx],

        if len(sbatch[0]) <= plus:
            fidx = bidx
            continue

        shuffleidx = np.arange(0, len(sbatch))
        if shuffle:
            np.random.shuffle(shuffleidx)
        sbatch = [sbatch[idx] for idx in shuffleidx]
        tbatch = [tbatch[idx] for idx in shuffleidx]
        sbatch_user = [sbatch_user[idx] for idx in shuffleidx]
        tbatch_user = [tbatch_user[idx] for idx in shuffleidx]  ##[batch, turns, length]
        sbatch_turn_id = [sbatch_turn_id[idx] for idx in shuffleidx]
        tbatch_turn_id = [tbatch_turn_id[idx] for idx in shuffleidx]
        assert len(sbatch[0]) <= len(tbatch[0])

        # convert to [turns, batch, lengths], [turns, batch, lengths]

        # pad src by turns
        # create the lengths: [turns, batch] for sbatch
        # sbatch: [turns, batch, lengths]
        pad_src_turn_num = len(tbatch[0]) - len(sbatch[0])
        if pad_src_turn_num > 0:
            sbatch_ = [[src_w2idx['<sos>'], PAD_U_ID, src_w2idx['<eos>']] for i in range(pad_src_turn_num)]
            id_max = max(tbatch_turn_id[0])
            sbatch_turn_id_ = [tbatch_turn_id[0][i] for i in range(pad_src_turn_num)]
            user_id = sbatch_user[0][0]
            sbatch_user_ = [user_id for i in range(pad_src_turn_num)]
            for i in range(len(sbatch)):
                sbatch[i] = sbatch_ + sbatch[i]
                sbatch_turn_id[i] = sbatch_turn_id_ + sbatch_turn_id[i]
                sbatch_user[i] = sbatch_user_ + sbatch_user[i]

        sbatch_turn_id = [tmp[::-1] for tmp in sbatch_turn_id]
        tbatch_turn_id = [tmp[::-1] for tmp in tbatch_turn_id]

        src_turn_lengths = []
        for i in range(len(sbatch)):
            lengths = []
            for item in sbatch[i]:
                lengths.append(len(item))
            src_turn_lengths.append(lengths)

        sbatch_ = []
        sbatch_turn_id_ = []
        for i in range(len(sbatch)):
            sent = []
            turn_id = []
            for j in range(len(sbatch[i])):
                sent += sbatch[i][j]
                turn_id += ([sbatch_turn_id[i][j]] * len(sbatch[i][j]))
            sbatch_.append(sent)
            sbatch_turn_id_.append(turn_id)
        sbatch = sbatch_
        sbatch_turn_id = sbatch_turn_id_
        pad_sequence(src_w2idx['<pad>'], sbatch, len(sbatch), maxlen=None)
        pad_sequence(sbatch_turn_id[-1][-1], sbatch_turn_id, len(sbatch_turn_id), maxlen=None)
        assert len(sbatch[0]) == len(sbatch_turn_id[0])

        tgt_turn_lengths = []
        for i in range(len(tbatch)):
            lengths = []
            for item in tbatch[i]:
                lengths.append(len(item))
            tgt_turn_lengths.append(lengths)
        # maxlen_tbatch = np.max(tgt_turn_lengths)
        # for i in range(len(tbatch)):
        #     pad_sequence(tgt_w2idx['<pad>'], tbatch[i], len(tbatch[i]), maxlen=None)

        # convert to tensor, change to cuda version tensor

        # sbatch [batch, turns_src*lengths]
        srcbatch = []
        for i in range(len(sbatch)):
            pause = torch.tensor(sbatch[i], dtype=torch.long)
            if torch.cuda.is_available():
                pause = pause.cuda()
            srcbatch.append(pause)
        sbatch = torch.stack(srcbatch)

        tgtbatch = []
        tbatch = transformer_list(tbatch)
        bs, ts = len(tbatch[0]), len(tbatch)
        # tbatch [batch, turns_tgt, lengths]
        for i in range(len(tbatch)):
            pad_sequence(tgt_w2idx['<pad>'], tbatch[i], len(tbatch[i]), maxlen=None) # None tgt_maxlen
            pause = torch.tensor(tbatch[i], dtype=torch.long)
            if torch.cuda.is_available():
                pause = pause.cuda()
            tgtbatch.append(pause)
        tbatch = tgtbatch # torch.stack(tgtbatch)

        src_turn_lengths = torch.tensor(src_turn_lengths, dtype=torch.long)
        tgt_turn_lengths = torch.tensor(tgt_turn_lengths, dtype=torch.long)
        sbatch_user = torch.tensor(sbatch_user, dtype=torch.long)  # [batch, turns_src]
        tbatch_user = torch.tensor(tbatch_user, dtype=torch.long)  # [batch, turns_tgt]
        sbatch_turn_id = torch.tensor(sbatch_turn_id, dtype=torch.long)
        tbatch_turn_id = torch.tensor(tbatch_turn_id, dtype=torch.long)

        sents_mapping, sent_rank, _ = src_doc_sents_map(sbatch)
        if torch.cuda.is_available():
            src_turn_lengths = src_turn_lengths.cuda()
            tgt_turn_lengths = tgt_turn_lengths.cuda()
            sbatch_turn_id = sbatch_turn_id.cuda()
            tbatch_turn_id = tbatch_turn_id.cuda()
            sents_mapping = sents_mapping.cuda()
            sent_rank = sent_rank.cuda()

        yield (sbatch, sents_mapping, sent_rank), tbatch, \
              (src_turn_lengths, tgt_turn_lengths,
               sbatch_user, tbatch_user,
               sbatch_turn_id, tbatch_turn_id)







if __name__ == "__main__":
    batch_num = 0
    src_w2idx, src_idx2w = load_pickle('./processed/dailydialog/iptvocab.pkl')
    tgt_w2idx, tgt_idx2w = load_pickle('./processed/dailydialog/optvocab.pkl')
    torch.cuda.set_device(2)
    for sbatch, tbatch, turn_lengths in get_batch_data_flatten('./data/dailydialog/src-train.txt', 
            './data/dailydialog/tgt-train.txt',
                                         './processed/dailydialog/iptvocab.pkl',
                                         './processed/dailydialog/optvocab.pkl',
                                         32, 50):
        ipdb.set_trace()
        print(len(sbatch), tbatch.shape, turn_lengths.shape)
        # if len(sbatch) == 3:
        #     ipdb.set_trace()
        batch_num += 1
    print('Batch_num:', batch_num)
