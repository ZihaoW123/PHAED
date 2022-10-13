#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.9.14

'''
utils function for training the model
'''

import numpy as np
import argparse
from collections import Counter
import pickle
import os
import re
import torch
import ipdb
import random
from tqdm import tqdm
from scipy.linalg import norm
try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
except:
    print(f'[!] cannot load module sklearn, ignore it')
from transformers import BertTokenizer

import nltk
from nltk.util import bigrams
from nltk.util import pad_sequence
from nltk.lm.preprocessing import pad_both_ends
from nltk.util import everygrams
from nltk.lm.preprocessing import flatten
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, Lidstone


def clean(s):
    # this pattern are defined for cleaning the dailydialog dataset
    s = s.strip().lower()
    s = re.sub(r'(\w+)\.(\w+)', r'\1 . \2', s)
    s = re.sub(r'(\w+)-(\w+)', r'\1 \2', s)
    # s = re.sub(r'[0-9]+(\.[0-9]+)?', r'1', s)
    s = s.replace('。', '.')
    # s = s.replace(';', ',')
    s = s.replace('...', ',')
    s = s.replace(' p . m . ', ' pm ')
    s = s.replace(' P . m . ', ' pm ')
    s = s.replace(' a . m . ', ' am ')
    
    # this pattern are defined for cleaning the ubuntu dataset
    # ....
    return s


# ========== calculate the N-gram perplexity ========== #
def train_ngram_lm(dataset, data, ngram=3, gamma=0.5):
    print(f'[!] max 3-gram, Lidstone smoothing with gamma 0.5')
    train, vocab = padded_everygram_pipeline(ngram, data)
    lm = Lidstone(gamma, ngram)
    lm.fit(train, vocab)
    with open(f'./data/{dataset}/lm.pkl', 'wb') as f:
        pickle.dump(lm, f)
    print(f'[!] ngram language model saved into ./data/{dataset}/lm.pkl')
    

def ngram_ppl(lm, test):
    return lm.perplexity(test)


# ========== jaccard, cosine + tf, cosine + tf-idf, GloVe ========== #
# ========== refer to: https://blog.csdn.net/asd991936157/article/details/77011206 ========== #
def jaccard_similarity(s1, s2):
    """
    :param s1: 
    :param s2: 
    :return: 
    """
    vectors = np.array([s1, s2])
    numerator = np.sum(np.min(vectors, axis=0))
    denominator = np.sum(np.max(vectors, axis=0))
    return 1.0 * numerator / denominator


def cosine_similarity_tf(s1, s2):
    """
    :param s1: 
    :param s2: 
    :return: 
    """
    return np.dot(s1, s2) / (norm(s1) * norm(s2))


def cosine_similarity_tfidf(s1, s2):
    """
    :param s1: 
    :param s2: 
    :return: 
    """
    return np.dot(s1, s2) / (norm(s2) * norm(s2))


def load_glove_embedding(path, dimension=300, lang='en'):
    if lang == 'en':
        exist_ = './data/glove_embedding.pkl'
    elif lang == 'zh':
        exist_ = './data/chinese_embedding.pkl'
    if os.path.exists(exist_):
        print(f'[!] load from the preprocessed embeddings {exist_}')
        return load_pickle(exist_)
    with open(path) as f:
        vocab = {}
        for line in tqdm(f.readlines()):
            line = line.split()
            assert len(line) > 300
            vector = np.array(list(map(float, line[-300:])), dtype=np.float)    # [300]
            vocab[line[0]] = vector
    vocab['<unk>'] = np.random.rand(dimension)
    print(f'[!] load word embedding from {path}')
    with open(exist_, 'wb') as f:
        pickle.dump(vocab, f)
    return vocab


def sent2glove(vocab, sent):
    s = np.zeros(vocab['<unk>'].shape, dtype=np.float)
    for word in nltk.word_tokenize(sent):
        # ipdb.set_trace()
        vector = vocab.get(word, vocab['<unk>'])
        s += vector
    return s

# ================================================================================= #
    

def load_best_model(dataset, model, net, min_threshold, max_threshold):
    path = f'./ckpt/{dataset}/{model}/'
    best_loss, best_file, best_epoch = np.inf, None, -1

    for file in os.listdir(path):
        _, val_loss, _, epoch = file.split('_')[:4]
        epoch = epoch.split('.')[0]
        val_loss, epoch = float(val_loss), int(epoch)

        if min_threshold <= epoch <= max_threshold and epoch > best_epoch:
            best_file = file
            best_epoch = epoch

    if best_file:
        file_path = path + best_file
        print(f'[!] Load the model from {file_path}, threshold ({min_threshold}, {max_threshold})')
        net.load_state_dict(torch.load(file_path)['net'])
    else:
        raise Exception('[!] No saved model')
        

def cos_similarity(gr, ge):
    # word embedding
    return np.dot(gr, ge) / (np.linalg.norm(gr) * np.linalg.norm(ge))


def num2seq(src, idx2w):
    # number to word sequence, src: [maxlen]
    return [idx2w[int(i)] for i in src]


def transformer_list(obj):
    # transformer [batch, turns, lengths] into [turns, batch, lengths]
    # turns are all the same for each batch
    turns = []
    batch_size, turn_size = len(obj), len(obj[0])
    for i in range(turn_size):
        turns.append([obj[j][i] for j in range(batch_size)])    # [batch, lengths]
    return turns


def pad_sequence(pad, batch, bs, maxlen=None):
    if maxlen is None:
        maxlen = max([len(batch[i]) for i in range(bs)])
    else:
        maxlen_ = max([len(batch[i]) for i in range(bs)])
        if maxlen < maxlen_:
            print('maxlen:', maxlen)
            print('maxlen_:', maxlen_)
        assert maxlen >= maxlen_

    for i in range(bs):
        batch[i].extend([pad] * (maxlen - len(batch[i])))


def load_pickle(file):
    with open(file, 'rb') as f:
        obj = pickle.load(f)
    return obj

PAD = '<pad>'
PAD_ID = 0
SOS = '<sos>'
SOS_ID = 1
EOS = '<eos>'
EOS_ID = 2
UNK = '<unk>'
UNK_ID = 3
PAD_U = '<pad_utterance>'
PAD_U_ID = 4
def generate_vocab(files, vocab, cutoff=50000):
    # training and validation files, input vocab and output vocab file
    words = []
    for file in files:
        with open(file, encoding='UTF-8') as f:
            for line in tqdm(f.readlines()):
                line = clean(line)
                list_words = nltk.word_tokenize(line)
                words.extend(list_words)
    words = Counter(words)
    print(f'[!] whole vocab size: {len(words)}')
    words = words.most_common(cutoff)
    # special token
    words_ = [(PAD, 1),
              (SOS, 1),
              (EOS, 1),
              (UNK, 1),
              (PAD_U, 1)]
    words_.extend(words)
    w2idx = {item[0]: idx for idx, item in enumerate(words_)}
    idx2w = [item[0] for item in words_]
    with open(vocab, 'wb') as f:
        pickle.dump((w2idx, idx2w), f)
    print(f'[!] Save the vocab into {vocab}, vocab_size: {len(w2idx)}')


def generate_bert_embedding(vocab, path):
    bc = BertClient()
    w2idx, idx2w = vocab
    words = [word for word in w2idx]
    emb = bc.encode(words)    # [vocab_size, 768], ndarray

    # save into the processed folder
    with open(path, 'wb') as f:
        pickle.dump(emb, f)

    print(f'[!] write the bert embedding into {path}')
    
    
# load data function for hierarchical models
def load_data(src, tgt, src_vocab, tgt_vocab, maxlen, tgt_maxlen, ld=True):
    # ld: whether load directly (VHRED/KgCVAE False), the target vocab of VHRED-based model is different from the original models, so the processed dataset isn't compatible.
    # convert dataset into src: [datasize, turns, lengths]
    # convert dataset into tgt: [datasize, lengths]
    # check the file, exist -> ignore
    # move it to the file `data_loader.py`
    src_prepath = os.path.splitext(src)[0] + '-hier.pkl'
    tgt_prepath = os.path.splitext(tgt)[0] + '-hier.pkl'
    if ld and os.path.exists(src_prepath) and os.path.exists(tgt_prepath):
        print(f'[!] preprocessed file {src_prepath} exist, load directly')
        print(f'[!] preprocessed file {tgt_prepath} exist, load directly')
        with open(src_prepath, 'rb') as f:
            src_dataset, src_user = pickle.load(f)
        with open(tgt_prepath, 'rb') as f:
            tgt_dataset, tgt_user = pickle.load(f)
        return src_dataset, src_user, tgt_dataset, tgt_user
    else:
        print(f'[!] cannot find the preprocessed file')
    
    src_w2idx, src_idx2w = load_pickle(src_vocab)
    tgt_w2idx, tgt_idx2w = load_pickle(tgt_vocab)
    src_user, tgt_user = [], []
    user_vocab = ['user0', 'user1']
    
    # src 
    with open(src, encoding='UTF-8') as f:
        src_dataset = []
        for line in tqdm(f.readlines()):
            line = clean(line)
            utterances = line.split('__eou__')    # only for chinese (zh50)
            turn = []
            srcu = []
            for utterance in utterances:
                if '<user0>' in utterance: user_c, user_cr = '<user0>', 'user0'
                elif '<user1>' in utterance: user_c, user_cr = '<user1>', 'user1'
                utterance = utterance.replace(user_c, user_cr).strip()
                line = [src_w2idx['<sos>']] + [src_w2idx.get(w, src_w2idx['<unk>']) for w in nltk.word_tokenize(utterance)] + [src_w2idx['<eos>']]
                if len(line) > maxlen:
                    line = [src_w2idx['<sos>'], line[1]] + line[-maxlen:]
                turn.append(line)
                srcu.append(user_vocab.index(user_cr))
            src_dataset.append(turn)
            src_user.append(srcu)

    # tgt
    with open(tgt, encoding='UTF-8') as f:
        tgt_dataset = []
        for line in tqdm(f.readlines()):
            line = clean(line)
            if '<user0>' in line: user_c, user_cr = '<user0>', 'user0'
            elif '<user1>' in line: user_c, user_cr = '<user1>', 'user1'
            line = line.replace(user_c, user_cr).strip()
            line = [tgt_w2idx['<sos>']] + [tgt_w2idx.get(w, tgt_w2idx['<unk>']) for w in nltk.word_tokenize(line)] + [tgt_w2idx['<eos>']]
            if len(line) > tgt_maxlen:
                line = line[:tgt_maxlen] + [tgt_w2idx['<eos>']]
            tgt_dataset.append(line)
            tgt_user.append(user_vocab.index(user_cr))
    
    if ld:
        with open(src_prepath, 'wb') as f:
            pickle.dump((src_dataset, src_user), f)
        with open(tgt_prepath, 'wb') as f:
            pickle.dump((tgt_dataset, tgt_user), f)
        print(f'[!] load dataset over, write into file {src_prepath} and {tgt_prepath}')
    else:
        print('[!] VHRED or KgCVAE donot write the dataset file')
 
    # src_user: [datasize, turn], tgt_user: [datasize]
    return src_dataset, src_user, tgt_dataset, tgt_user


def load_data_users(src, tgt, src_vocab, tgt_vocab, maxlen, tgt_maxlen, ld=True):
    # ld: whether load directly (VHRED/KgCVAE False), the target vocab of VHRED-based model is different from the original models, so the processed dataset isn't compatible.
    # convert dataset into src: [datasize, turns, lengths]
    # convert dataset into tgt: [datasize, lengths]
    # check the file, exist -> ignore
    # move it to the file `data_loader.py`
    src_prepath = os.path.splitext(src)[0] + '-hier-users.pkl'
    tgt_prepath = os.path.splitext(tgt)[0] + '-hier-users.pkl'
    if ld and os.path.exists(src_prepath) and os.path.exists(tgt_prepath):
        print(f'[!] preprocessed file {src_prepath} exist, load directly')
        print(f'[!] preprocessed file {tgt_prepath} exist, load directly')
        with open(src_prepath, 'rb') as f:
            src_dataset, src_user, src_turns_num = pickle.load(f)
        with open(tgt_prepath, 'rb') as f:
            tgt_dataset, tgt_user, tgt_turns_num = pickle.load(f)
        return src_dataset, tgt_dataset, src_user, tgt_user, src_turns_num, tgt_turns_num
    else:
        print(f'[!] cannot find the preprocessed file')

    src_w2idx, src_idx2w = load_pickle(src_vocab)
    tgt_w2idx, tgt_idx2w = load_pickle(tgt_vocab)
    src_user, tgt_user = [], []
    user_vocab = ['user0', 'user1']

    src_dataset = []
    tgt_dataset = []
    src_turns_num = []
    tgt_turns_num = []
    # tgt_user
    # tgt
    tgt_user_ =[]
    with open(tgt, encoding='UTF-8') as f:
        for line in tqdm(f.readlines()):
            line = clean(line)
            if '<user0>' in line:
                user_c, user_cr = '<user0>', 'user0'
            elif '<user1>' in line:
                user_c, user_cr = '<user1>', 'user1'
            tgt_user_.append(user_vocab.index(user_cr))
    # src

    with open(src, encoding='UTF-8') as f:
        not_complete_src = 0
        complete_src = 0
        complete_tgt = 0
        not_complete_tgt1 = 0
        not_complete_tgt2 = 0
        for line in tqdm(f.readlines()):
            line = clean(line)
            utterances = line.strip().split('__eou__')  # only for chinese (zh50)\

            tgt_turn = []
            src_turn = []
            tgt_turn_num = []
            src_turn_num = []
            srcu = []
            tgtu = []

            tgt_user_id = tgt_user_[len(src_dataset)]

            num_turn = int(len(utterances))
            for utterance in utterances:
                if '<user0>' in utterance:
                    user_c, user_cr = '<user0>', 'user0'
                elif '<user1>' in utterance:
                    user_c, user_cr = '<user1>', 'user1'
                utterance = utterance.replace(user_c, user_cr).strip()
                line = [src_w2idx['<sos>']] + [src_w2idx.get(w, src_w2idx['<unk>']) for w in
                                               nltk.word_tokenize(utterance)] + [src_w2idx['<eos>']]

                if tgt_user_id != user_vocab.index(user_cr):
                    if len(line) > maxlen:
                        line = [src_w2idx['<sos>'], line[1]] + line[-(maxlen-2):]
                        not_complete_src += 1
                    else:
                        complete_src += 1
                    src_turn.append(line)
                    srcu.append(user_vocab.index(user_cr))
                    src_turn_num.append(num_turn-1)
                else:
                    len_line = len(line)
                    line_old = line[:]
                    if len_line > tgt_maxlen:
                        line = line[:tgt_maxlen-1]
                        len_line = len(line)
                        line_re = line[:]
                        line_re.reverse()
                        end_1 = len_line - line_re.index(tgt_w2idx['.']) if tgt_w2idx['.'] in line_re else -1
                        end_2 = len_line - line_re.index(tgt_w2idx['?']) if tgt_w2idx['?'] in line_re else -1
                        end_3 = len_line - line_re.index(tgt_w2idx['!']) if tgt_w2idx['!'] in line_re else -1
                        end = max(end_1, end_2, end_3)
                        if end == -1:
                            end = len_line - line_re.index(tgt_w2idx[',']) if tgt_w2idx[','] in line_re else -1
                            if end == -1:
                                end = tgt_maxlen-1
                                # print('no , in sent')
                            line = line[:end-1] + [tgt_w2idx['.'], tgt_w2idx['<eos>']]
                            # print('not complete sentense:', len(line_old), ' >',num2seq(line_old, tgt_idx2w))
                            assert len(line) <= tgt_maxlen
                            not_complete_tgt2 += 1
                        else:
                            line = line[:end] + [tgt_w2idx['<eos>']]
                            assert len(line) <= tgt_maxlen
                            not_complete_tgt1 += 1
                        
                    else:
                        complete_tgt += 1
                    tgt_turn.append(line)
                    tgtu.append(user_vocab.index(user_cr))
                    tgt_turn_num.append(num_turn-1)
                num_turn = num_turn - 1
            for t in range(len(tgt_turn_num)-1):
                cha = tgt_turn_num[t] - tgt_turn_num[t+1]
                if cha > 1:
                    tgt_turn_num[:t+1] = [x-(cha-1) for x in tgt_turn_num[:t+1]]
            for t in range(len(src_turn_num)-1):
                cha = src_turn_num[t] - src_turn_num[t+1]
                if cha > 1:
                    src_turn_num[:t+1] = [x-(cha-1) for x in src_turn_num[:t+1]]

            src_dataset.append(src_turn)
            tgt_dataset.append(tgt_turn)
            src_user.append(srcu)
            tgt_user.append(tgtu)
            src_turns_num.append(src_turn_num)
            tgt_turns_num.append(tgt_turn_num)
    # tgt
    tgt_line_n = 0
    with open(tgt, encoding='UTF-8') as f:
        for line in tqdm(f.readlines()):
            line = clean(line)
            if '<user0>' in line:
                user_c, user_cr = '<user0>', 'user0'
            elif '<user1>' in line:
                user_c, user_cr = '<user1>', 'user1'
            line = line.replace(user_c, user_cr).strip()
            line = [tgt_w2idx['<sos>']] + [tgt_w2idx.get(w, tgt_w2idx['<unk>']) for w in nltk.word_tokenize(line)] + [
                tgt_w2idx['<eos>']]
            len_line = len(line)
            line_old = line[:]
            if len(line) > tgt_maxlen:
                line = line[:tgt_maxlen-1]
                len_line = len(line)
                line_re = line[:]
                line_re.reverse()
                end_1 = len_line - line_re.index(tgt_w2idx['.']) if tgt_w2idx['.'] in line_re else -1
                end_2 = len_line - line_re.index(tgt_w2idx['?']) if tgt_w2idx['?'] in line_re else -1
                end_3 = len_line - line_re.index(tgt_w2idx['!']) if tgt_w2idx['!'] in line_re else -1
                end = max(end_1, end_2, end_3)
                if end == -1:
                    end = len_line - line_re.index(tgt_w2idx[',']) if tgt_w2idx[','] in line_re else -1
                    if end == -1:
                        end = tgt_maxlen-1
                        # print('no , in sent')
                    line = line[:end-1] + [tgt_w2idx['.'], tgt_w2idx['<eos>']]
                    # print('not complete sentense:', len(line_old), num2seq(line_old, tgt_idx2w))
                    not_complete_tgt2 += 1
                else:
                    line = line[:end] + [tgt_w2idx['<eos>']]
                    not_complete_tgt1 += 1
            else:
                complete_tgt += 1 
            tgt_dataset[tgt_line_n].append(line)
            tgt_user[tgt_line_n].append(user_vocab.index(user_cr))
            tgt_turns_num[tgt_line_n].append(0)
            tgt_line_n += 1
    print('complete_tgt', complete_tgt, 'not_complete_tgt1', not_complete_tgt1, 'not_complete_tgt2', not_complete_tgt2)
    print('complete_src', complete_src, 'not_complete_src', not_complete_src)
    assert tgt_line_n == len(src_dataset)
    if ld:
        with open(src_prepath, 'wb') as f:
            pickle.dump((src_dataset, src_user, src_turns_num), f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(tgt_prepath, 'wb') as f:
            pickle.dump((tgt_dataset, tgt_user, tgt_turns_num), f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'[!] load dataset over, write into file {src_prepath} and {tgt_prepath}')
    else:
        print('[!] VHRED or KgCVAE donot write the dataset file')

    # src_user: [datasize, turn], tgt_user: [datasize]
    return src_dataset, tgt_dataset, src_user, tgt_user, src_turns_num, tgt_turns_num
    


def load_data_users_wo_userid(src, tgt, src_vocab, tgt_vocab, maxlen, tgt_maxlen, ld=True):
    # ld: whether load directly (VHRED/KgCVAE False), the target vocab of VHRED-based model is different from the original models, so the processed dataset isn't compatible.
    # convert dataset into src: [datasize, turns, lengths]
    # convert dataset into tgt: [datasize, lengths]
    # check the file, exist -> ignore
    # move it to the file `data_loader.py`
    src_prepath = os.path.splitext(src)[0] + '-hier-users-wo-userid.pkl'
    tgt_prepath = os.path.splitext(tgt)[0] + '-hier-users-wo-userid.pkl'
    if ld and os.path.exists(src_prepath) and os.path.exists(tgt_prepath):
        print(f'[!] preprocessed file {src_prepath} exist, load directly')
        print(f'[!] preprocessed file {tgt_prepath} exist, load directly')
        with open(src_prepath, 'rb') as f:
            src_dataset, src_user, src_turns_num = pickle.load(f)
        with open(tgt_prepath, 'rb') as f:
            tgt_dataset, tgt_user, tgt_turns_num = pickle.load(f)
        return src_dataset, tgt_dataset, src_user, tgt_user, src_turns_num, tgt_turns_num
    else:
        print(f'[!] cannot find the preprocessed file')

    src_w2idx, src_idx2w = load_pickle(src_vocab)
    tgt_w2idx, tgt_idx2w = load_pickle(tgt_vocab)
    src_user, tgt_user = [], []
    user_vocab = ['user0', 'user1']

    src_dataset = []
    tgt_dataset = []
    src_turns_num = []
    tgt_turns_num = []
    # tgt_user
    # tgt
    tgt_user_ =[]
    with open(tgt, encoding='UTF-8') as f:
        for line in tqdm(f.readlines()):
            line = clean(line)
            if '<user0>' in line:
                user_c, user_cr = '<user0>', 'user0'
            elif '<user1>' in line:
                user_c, user_cr = '<user1>', 'user1'
            tgt_user_.append(user_vocab.index(user_cr))
    # src

    with open(src, encoding='UTF-8') as f:
        not_complete_src = 0
        complete_src = 0
        complete_tgt = 0
        not_complete_tgt1 = 0
        not_complete_tgt2 = 0
        for line in tqdm(f.readlines()):
            line = clean(line)
            utterances = line.strip().split('__eou__')  # only for chinese (zh50)\

            tgt_turn = []
            src_turn = []
            tgt_turn_num = []
            src_turn_num = []
            srcu = []
            tgtu = []

            tgt_user_id = tgt_user_[len(src_dataset)]

            num_turn = int(len(utterances))
            for utterance in utterances:
                if '<user0>' in utterance:
                    user_c, user_cr = '<user0>', 'user0'
                elif '<user1>' in utterance:
                    user_c, user_cr = '<user1>', 'user1'
                utterance = utterance.replace(user_c, '').strip()
                line = [src_w2idx['<sos>']] + [src_w2idx.get(w, src_w2idx['<unk>']) for w in
                                               nltk.word_tokenize(utterance)] + [src_w2idx['<eos>']]

                if tgt_user_id != user_vocab.index(user_cr):
                    if len(line) > maxlen:
                        line = [src_w2idx['<sos>'], line[1]] + line[-(maxlen-2):]
                        not_complete_src += 1
                    else:
                        complete_src += 1
                    src_turn.append(line)
                    srcu.append(user_vocab.index(user_cr))
                    src_turn_num.append(num_turn-1)
                else:
                    len_line = len(line)
                    line_old = line[:]
                    if len_line > tgt_maxlen:
                        line = line[:tgt_maxlen-1]
                        len_line = len(line)
                        line_re = line[:]
                        line_re.reverse()
                        end_1 = len_line - line_re.index(tgt_w2idx['.']) if tgt_w2idx['.'] in line_re else -1
                        end_2 = len_line - line_re.index(tgt_w2idx['?']) if tgt_w2idx['?'] in line_re else -1
                        end_3 = len_line - line_re.index(tgt_w2idx['!']) if tgt_w2idx['!'] in line_re else -1
                        end = max(end_1, end_2, end_3)
                        if end == -1:
                            end = len_line - line_re.index(tgt_w2idx[',']) if tgt_w2idx[','] in line_re else -1
                            if end == -1:
                                end = tgt_maxlen-1
                                print('no , in sent')
                            line = line[:end-1] + [tgt_w2idx['.'], tgt_w2idx['<eos>']]
                            # print('not complete sentense:', len(line_old), ' >',num2seq(line_old, tgt_idx2w))
                            assert len(line) <= tgt_maxlen
                            not_complete_tgt2 += 1
                        else:
                            line = line[:end] + [tgt_w2idx['<eos>']]
                            assert len(line) <= tgt_maxlen
                            not_complete_tgt1 += 1
                        
                    else:
                        complete_tgt += 1
                    tgt_turn.append(line)
                    tgtu.append(user_vocab.index(user_cr))
                    tgt_turn_num.append(num_turn-1)
                num_turn = num_turn - 1
            for t in range(len(tgt_turn_num)-1):
                cha = tgt_turn_num[t] - tgt_turn_num[t+1]
                if cha > 1:
                    tgt_turn_num[:t+1] = [x-(cha-1) for x in tgt_turn_num[:t+1]]
            for t in range(len(src_turn_num)-1):
                cha = src_turn_num[t] - src_turn_num[t+1]
                if cha > 1:
                    src_turn_num[:t+1] = [x-(cha-1) for x in src_turn_num[:t+1]]

            src_dataset.append(src_turn)
            tgt_dataset.append(tgt_turn)
            src_user.append(srcu)
            tgt_user.append(tgtu)
            src_turns_num.append(src_turn_num)
            tgt_turns_num.append(tgt_turn_num)
    # tgt
    tgt_line_n = 0
    with open(tgt, encoding='UTF-8') as f:
        for line in tqdm(f.readlines()):
            line = clean(line)
            if '<user0>' in line:
                user_c, user_cr = '<user0>', 'user0'
            elif '<user1>' in line:
                user_c, user_cr = '<user1>', 'user1'
            line = line.replace(user_c, '').strip()
            line = [tgt_w2idx['<sos>']] + [tgt_w2idx.get(w, tgt_w2idx['<unk>']) for w in nltk.word_tokenize(line)] + [
                tgt_w2idx['<eos>']]
            len_line = len(line)
            line_old = line[:]
            if len(line) > tgt_maxlen:
                line = line[:tgt_maxlen-1]
                len_line = len(line)
                line_re = line[:]
                line_re.reverse()
                end_1 = len_line - line_re.index(tgt_w2idx['.']) if tgt_w2idx['.'] in line_re else -1
                end_2 = len_line - line_re.index(tgt_w2idx['?']) if tgt_w2idx['?'] in line_re else -1
                end_3 = len_line - line_re.index(tgt_w2idx['!']) if tgt_w2idx['!'] in line_re else -1
                end = max(end_1, end_2, end_3)
                if end == -1:
                    end = len_line - line_re.index(tgt_w2idx[',']) if tgt_w2idx[','] in line_re else -1
                    if end == -1:
                        end = tgt_maxlen-1
                        print('no , in sent')
                    line = line[:end-1] + [tgt_w2idx['.'], tgt_w2idx['<eos>']]
                    # print('not complete sentense:', len(line_old), num2seq(line_old, tgt_idx2w))
                    not_complete_tgt2 += 1
                else:
                    line = line[:end] + [tgt_w2idx['<eos>']]
                    not_complete_tgt1 += 1
            else:
                complete_tgt += 1 
            tgt_dataset[tgt_line_n].append(line)
            tgt_user[tgt_line_n].append(user_vocab.index(user_cr))
            tgt_turns_num[tgt_line_n].append(0)
            tgt_line_n += 1
    print('complete_tgt', complete_tgt, 'not_complete_tgt1', not_complete_tgt1, 'not_complete_tgt2', not_complete_tgt2)
    print('complete_src', complete_src, 'not_complete_src', not_complete_src)
    assert tgt_line_n == len(src_dataset)
    if ld:
        with open(src_prepath, 'wb') as f:
            pickle.dump((src_dataset, src_user, src_turns_num), f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(tgt_prepath, 'wb') as f:
            pickle.dump((tgt_dataset, tgt_user, tgt_turns_num), f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'[!] load dataset over, write into file {src_prepath} and {tgt_prepath}')
    else:
        print('[!] VHRED or KgCVAE donot write the dataset file')

    # src_user: [datasize, turn], tgt_user: [datasize]
    return src_dataset, tgt_dataset, src_user, tgt_user, src_turns_num, tgt_turns_num

def create_the_abs_graph(turns, weights=[1, 1], threshold=1, bidir=False, self_loop=False):
    '''
    empchat: fully connected network
    '''
    edges = {}
    s_w, u_w = weights
    turn_len = len(turns)
    for i in range(turn_len):
        for j in range(turn_len):
            if i == j:
                if self_loop:
                    edges[(i, j)] = [s_w]
            else:
                edges[(i, j)] = [s_w]
                
    # clean the edges
    e, w = [[], []], []
    whole_num = 0
    for src, tgt in edges.keys():
        e[0].append(src)
        e[1].append(tgt)
        w.append(max(edges[(src, tgt)]))
        whole_num += 1
        
        if bidir and src != tgt:
            e[0].append(tgt)
            e[1].append(src)
            w.append(max(edges[(src, tgt)]))
            whole_num += 1
            
    #  print(f'[!] whole edges number: {whole_num}')
    return (e, w), whole_num
    
    
def create_the_graph(turns, vocab, weights=[1, 1], threshold=0.8, bidir=False):
    '''create the weighted directed graph of one conversation
    sequenutial edge, user connected edge, [BERT/PMI] edge
    param: turns: [turns(user, utterance)]
    param: weights: [sequential_w, user_w]
    output: [2, num_edges], [num_edges]
    
    For dataset DSTC7, [sequential edges, last_utterence edges, user edges, self-loop]
    
    For Dailydialog dataset, [sequentail edge, first utterance edge, last utterence edges]
    
    For personachat dataset, [last utterence edges, correlation edges (threshold=0.8)]
    
    For ubuntu dataset, [seqential edges, user edges, last utterance edges, correlation edges (threshold=0.6)]
    
    For cornell dataset, [seqential edges, last utterance edges, correlation edges (threshold=0.6)]
    
    For empchat dataset, [sequentil edges, last utterance edges, user edges, self-loop]
    '''
    edges = {}
    s_w, u_w = weights
    # sequential edges, (turn_len - 1)
    turn_len = len(turns)
    se, ue, pe = 0, 0, 0
    for i in range(turn_len - 1):
        edges[(i, i + 1)] = [s_w]
        se += 1
        
    '''
    # user edge
    for i in range(turn_len):
        for j in range(turn_len):
            if j > i:
                if 'user0' in turns[i]:
                    useri = 'user0'
                elif 'user1' in turns[i]:
                    useri = 'user1'
                else:
                    ipdb.set_trace()
                if 'user0' in turns[j]:
                    userj = 'user0'
                elif 'user1' in turns[j]:
                    userj = 'user1'
                else:
                    ipdb.set_trace()
                if useri == userj:
                    if edges.get((i, j), None):
                        edges[(i, j)].append(u_w)
                    else:
                        edges[(i, j)] = [u_w]
                    ue += 1
    '''
                    
    # ========== NOTE NOTE NOTE ========== #
    # all for the last query
    # NOTE: Remember to reverse the direction
    query = turn_len-1
    for i in range(turn_len):
        # if edges.get((i, query), None):
        if edges.get((query, i), None):
            # edges[(i, query)].append(u_w)
            edges[(query, i)].append(u_w)
        else:
            # edges[(i, query)] = [u_w]
            edges[(query, i)] = [u_w]
        if edges.get((i, query), None):
            edges[(i, query)].append(u_w)
        else:
            edges[(i, query)] = [u_w]
            
    query = 0
    for i in range(turn_len):
        # if edges.get((i, query), None):
        if edges.get((query, i), None):
            # edges[(i, query)].append(u_w)
            edges[(query, i)].append(u_w)
        else:
            # edges[(i, query)] = [u_w]
            edges[(query, i)] = [u_w]
        if edges.get((i, query), None):
            edges[(i, query)].append(u_w)
        else:
            edges[(i, query)] = [u_w]
           
    '''
    # distance
    utterances = []
    for utterance in turns:
        utterance = utterance.replace('user0', '').strip()
        utterance = utterance.replace('user1', '').strip()
        if utterance:
            utterances.append(utterance)
        else:
            utterances.append('<unk>')
            
    # ========== TFIDF, Counter, GloVe embedding ========== #
    count_vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize)
    count_vectors = count_vectorizer.fit_transform(utterances).toarray()    # [datasize, word_size]
    # print(f'[!] over the count fit_transform, shape {count_vectors.shape}')
    tfidf_vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
    tfidf_vectors = tfidf_vectorizer.fit_transform(utterances).toarray()    # [datasize, word_size]
    # print(f'[!] over the tfidf fit_transform, shape: {tfidf_vectors.shape}')
        
    # add the edges accorading to the TFIDF and Counter information
    for i in range(turn_len):
        for j in range(turn_len):
            if j > i:
                utter1, utter2 = count_vectors[i], count_vectors[j]
                # jaccard
                jaccard = jaccard_similarity(utter1, utter2)
                # cosine + tf
                cosine_tf = cosine_similarity_tf(utter1, utter2)
                # cosine + tfidf 
                cosine_tf_idf = cosine_similarity_tfidf(utter1, utter2)
                # glove embedding
                # utter1 = sent2glove(vocab, utterances[i])
                # utter2 = sent2glove(vocab, utterances[j])
                # glove = cos_similarity(utter1, utter2)
                
                weight = max([jaccard, cosine_tf, cosine_tf_idf])
                
                if weight >= threshold:
                    if edges.get((i, j), None):
                        edges[(i, j)].append(weight * u_w)
                    else:
                        edges[(i, j)] = [weight * u_w]
                    pe += 1
    '''
    

    # clean the edges
    e, w = [[], []], []
    whole_num = 0
    for src, tgt in edges.keys():
        e[0].append(src)
        e[1].append(tgt)
        w.append(max(edges[(src, tgt)]))
        whole_num += 1
        
        if bidir and src != tgt:
            e[0].append(tgt)
            e[1].append(src)
            w.append(max(edges[(src, tgt)]))
            whole_num += 1
            
    # print(f'[!] whole number edges is {whole_num}')
    return (e, w), whole_num


def generate_graph(dialogs, path, fully=False, threshold=0.75, 
                   bidir=False, lang='en', self_loop=False):
    # dialogs: [datasize, turns]
    # return: [datasize, (2, num_edges)/ (num_edges)]
    # **make sure the bert-as-service is running**
    edges = []
    sum_num = 0
    if lang == 'en':
        wbpath = '/home/lt/data/File/wordembedding/glove/glove.6B.300d.txt'
    elif lang == 'zh':
        wbpath = '/home/lt/data/File/wordembedding/chinese/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
    else:
        raise Exception(f'[!] unknown language of word embedding path {lang}')
    if not fully:
        print(f'[!] prepare to load the 300 embedding from {wbpath} (you can change this path)')
        vocab = load_glove_embedding(wbpath, lang=lang)
    else:
        print(f'[!] donot need the word embedding for constructing the graph')
    for dialog in tqdm(dialogs):
        if fully:
            edge, num_e = create_the_abs_graph(dialog, weights=[1, 1],
                                               threshold=threshold, 
                                               bidir=bidir, self_loop=self_loop)
        else:
            edge, num_e = create_the_graph(dialog, vocab, 
                                           threshold=threshold,
                                           bidir=bidir)
        sum_num += num_e
        edges.append(edge)

    with open(path, 'wb') as f:
        pickle.dump(edges, f)

    print(f'[!] avg edges number is {round(sum_num / len(dialogs), 4)}')
    print(f'[!] graph information is converted in {path}')


def idx2sent(data, vocab):
    # turn the index to the sentence
    # data: [datasize, turn, length]
    # user: [datasize, turn]
    # return: [datasize, (user, turns)]
    _, idx2w = load_pickle(vocab)
    datasets = []
    for example in tqdm(data):
        # example: [turn, length], user: [turn]
        turns = []
        for turn in example:
            utterance = ' '.join([idx2w[w] for w in turn])
            utterance = utterance.replace('<sos>', '').replace('<eos>', '').strip()
            turns.append(utterance)
        datasets.append(turns)
    return datasets

# ========== stst of the graph ========== #
def analyse_graph(path, hops=3):
    '''
    This function analyzes the graph coverage stat of the graph in Dailydialog 
    and cornell dataset.
    Stat the context node coverage of each node in the conversation.
    :param: path, the path of the dataset graph file.
    ''' 
    graph = load_pickle(path)    # [datasize, ([2, num_edge], [num_edge])]
    sum_graph, sum_in, sum_out = [], [], []
    for idx, (edges, _) in enumerate(tqdm(graph)):
        # make sure the number of the nodes
        sum_graph.append(len(edges[0]))
        # in degree
        sum_in_dict = {}
        for i in edges[1]:
            if i in sum_in_dict:
                sum_in_dict[i] += 1
            else:
                sum_in_dict[i] = 1
        sum_in.extend(list(sum_in_dict.values()))
        sum_out_dict = {}
        for i in edges[0]:
            if i in sum_out_dict:
                sum_out_dict[i] += 1
            else:
                sum_out_dict[i] = 1
        sum_out.extend(list(sum_out_dict.values()))
        
    # ========== stat ========== #
    avg_graph = np.mean(sum_graph)
    avg_in = np.mean(sum_in)
    avg_out = np.mean(sum_out)
    print(f'[!] the avg edges numbers in graph: {round(avg_graph, 4)}')
    print(f'[!] the avg in-degree numbers: {round(avg_in, 4)}')
    print(f'[!] the avg out_degree numbers: {round(avg_out, 4)}')
    
    
def Perturbations_test(src_test_in, src_test_out, mode=1):
    '''
    ACL 2019 Short paper:
    Do Neural Dialog Systems Use the Conversation History Effectively? An Empirical Study
    
    ## Utterance-level
    1. Shuf: shuffles the sequence of utterances in the dialog history
    2. Rev:  reverses the order of utterances in the history (but maintains word order within each utterance)
    3.4. Drop: completely drops certain utterances (drop first / drop last)
    5. Truncate: that truncates the dialog history
    
    ## Word-level
    6. word-shuffle: randomly shuffles the words within an utterance
    7. reverse: reverses the ordering of words
    8. word-drop: drops 30% of the words uniformly
    9. noun-drop: drops all nouns
    10. verb-drop: drops all verbs
    '''
    
    # load the file
    with open(src_test_in) as f:
        corpus = []
        for line in f.readlines():
            line = line.strip()
            sentences = line.split('__eou__')
            sentences = [i.strip() for i in sentences]
            corpus.append(sentences)
    print(f'[!] load the data from {src_test_in}')
    
    print(f'[!] perburtation mode: {mode}')
    # perturbation
    new_corpus = []
    for i in corpus:
        if mode == 1:
            random.shuffle(i)
            new_corpus.append(i) 
        elif mode == 2:
            new_corpus.append(list(reversed(i)))
        elif mode == 3:
            if len(i) > 1:
                new_corpus.append(i[1:])
            else:
                new_corpus.append(i)
        elif mode == 4:
            if len(i) > 1:
                new_corpus.append(i[:-1])
            else:
                new_corpus.append(i)
        elif mode == 5:
            new_corpus.append([i[-1]])
        elif mode == 6:
            s_ = []
            for s in i:
                user, s = s[:8], s[8:].strip()
                words = nltk.word_tokenize(s)
                random.shuffle(words)
                s_.append(user + ' '.join(words))
            new_corpus.append(s_)
        elif mode == 7:
            s_ = []
            for s in i:
                user, s = s[:8], s[8:].strip()
                words = nltk.word_tokenize(s)
                s_.append(user + ' '.join(list(reversed(words))))
            new_corpus.append(s_)
        elif mode == 8:
            s_ = []
            for s in i:
                user, s = s[:8], s[8:].strip()
                words = nltk.word_tokenize(s)
                words = [w_ for w_ in words if random.random() > 0.3]
                s_.append(user + ' '.join(words))
            new_corpus.append(s_)
        elif mode == 9:
            s_ = []
            for s in i:
                user, s = s[:8], s[8:].strip()
                tagger = nltk.pos_tag(nltk.word_tokenize(s))
                words = []
                for w, t in tagger:
                    if t in ['NN', 'NNS', 'NNP', 'NNPS']:
                        continue
                    else:
                        words.append(w)
                s_.append(user + ' '.join(words))
            new_corpus.append(s_)
        elif mode == 10:
            s_ = []
            for s in i:
                user, s = s[:8], s[8:].strip()
                tagger = nltk.pos_tag(nltk.word_tokenize(s))
                words = []
                for w, t in tagger:
                    if t in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                        continue
                    else:
                        words.append(w)
                s_.append(user + ' '.join(words))
            new_corpus.append(s_)
        else:
            raise Exception(f'[!] wrong mode: {mode}')
    
    # write the new source test file
    with open(src_test_out, 'w') as f:
        for i in new_corpus:
            i = ' __eou__ '.join(i)
            f.write(f'{i}\n')
    print(f'[!] write the new file into {src_test_out}')
    
    
def read_file(path):
    with open(path) as f:
        corpus = []
        for line in f.readlines():
            line = line.strip()
            corpus.append(line.split())
    return corpus


def read_pred_file(path):
    with open(path) as f:
        ref, tgt = [], []
        for idx, line in enumerate(f.readlines()):
            if idx % 4 == 1:
                line = line.replace("user1", "").replace("user0", "").replace("- ref: ", "").replace('<sos>', '').replace('<eos>', '').strip()
                ref.append(line.split())
            elif idx % 4 == 2:
                line = line.replace("user1", "").replace("user0", "").replace("- tgt: ", "").replace('<sos>', '').replace('<eos>', '').strip()
                tgt.append(line.split())
    # filter the empty line
    ref = [i for i in ref if i]
    tgt = [i for i in tgt if i]
    return ref, tgt


def analyse_coverage_word_embedding(vocab, lang='en'):
    if lang == 'en':
        wbpath = '/home/lt/data/File/wordembedding/glove/glove.6B.300d.txt'
    elif lang == 'zh':
        wbpath = '/home/lt/data/File/wordembedding/chinese/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
    else:
        raise Exception(f'[!] unknown language of word embedding path {lang}')
    count = 0
    nvocab = load_glove_embedding(wbpath, lang=lang)
    w2idx, idx2w = load_pickle(vocab)
    for word in idx2w:
        if word in nvocab.keys():
            count += 1
            
    print(f'[!] the coverage of the word embedding is {count}/{len(idx2w)}/{round(count / len(idx2w), 2)}')
    
    
def analyse_dataset(dataset, split):
    # analyse the dataset setting, adjust the padding lengths
    print(f'==========================================')
    print(f'[!] the metadata of {dataset}-src-{split}')
    words = set([])
    with open(f'./data/{dataset}/src-{split}.txt') as f:
        turn, tcounter = [], 0
        i, j, icounter, jcounter = 0, 0, 0, 0
        imax, imin, jmax, jmin = -10000, 10000, -10000, 10000
        for line in f.readlines():
            line = line.strip()
            j += len(line.split())
            words |= set(line.split())
            jcounter += 1
            jmin = min(jmin, len(line.split()))
            jmax = max(jmax, len(line.split()))
            lines = line.strip().split('__eou__')
            turn.append(len(lines))
            tcounter += 1
            for k in lines:
                i += len(k.split())
                icounter += 1
                imin = min(imin, len(k.split()))
                imax = max(imax, len(k.split()))
    print(f'[!] length of the sentenes(avg, max, min) for hierarchical: {round(i/icounter, 4)}/{imax}/{imin}')
    print(f'[!] length of the sentenes(avg, max, min) for no-hierarchical: {round(j/jcounter, 4)}/{jmax}/{jmin}')
    max_t, min_t, avg_t = max(turn), min(turn), np.mean(turn)
    print(f'[!] turn length(max/min/avg): {round(max_t, 4)}/{round(min_t, 4)}/{round(avg_t, 4)}')
    
    # responses 
    print(f'[!] the metadata of {dataset}-tgt-{split}')
    with open(f'./data/{dataset}/tgt-{split}.txt') as f:
        i, j, icounter, jcounter = 0, 0, 0, 0
        imax, imin, jmax, jmin = -10000, 10000, -10000, 10000
        for line in f.readlines():
            line = line.strip()
            j += len(line.split())
            jcounter += 1
            jmin = min(jmin, len(line.split()))
            jmax = max(jmax, len(line.split()))
    print(f'[!] length of the responses(avg, max, min): {round(j/jcounter, 4)}/{jmax}/{jmin}')
    
    print(f'[!] total words: {len(words)}')
    print(f'==========================================')
                
    
# ========== function for transformers (GPT2) ==========
def transformer_preprocess(src_path, tgt_path, tokenized_file, 
                           vocab_file='./config/vocab_en.txt', ctx=200):
    '''
    tokenize the dataset for NLG (GPT2), write the tokenized id into the tokenized_file.
    more details can be found in https://github.com/yangjianxin1/GPT2-chitchat
    '''
    def clean_inside(s):
        s = s.replace('<user0>', '')
        s = s.replace('<user1>', '')
        s = s.strip()
        s = clean(s)
        return s
        
    # create the Bert tokenizer of the GPT2 model
    tokenizer = BertTokenizer(vocab_file=vocab_file)
    
    src_data, tgt_data = read_file(src_path), read_file(tgt_path)
    src_data = [' '.join(i) for i in src_data]
    tgt_data = [' '.join(i) for i in tgt_data]
    assert len(src_data) == len(tgt_data), f'[!] length of src and tgt: {len(src_data)}/{len(tgt_data)}'
    
    # combine them
    corpus = []
    longest = 0
    for s, t in tqdm(list(zip(src_data, tgt_data))):
        item = [tokenizer.cls_token_id]   # [CLS] for each dialogue in the begining
        s = s + ' __eou__ ' + t
        s = clean_inside(s)
        utterances = s.split('__eou__')
        for utterance in utterances:
            words = nltk.word_tokenize(utterance)
            item.extend([tokenizer.convert_tokens_to_ids(word) for word in words])
            item.append(tokenizer.sep_token_id)
        if len(item) > longest:
            longest = len(item)
        item = item[:ctx]
        corpus.append(item)
        
    # write into the file
    with open(tokenized_file, 'w') as f:
        for i in range(len(corpus)):
            words = [str(word) for word in corpus[i]]
            f.write(f'{" ".join(words)}')
            if i < len(corpus) - 1:
                f.write('\n')
                
    print(f'[!] Preprocess the data for the transformers(GPT2), the longest sentence :{longest}, write the data into {tokenized_file}.')
    
    
# From https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py
# ========== lr scheduler for transformer ==========
class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps, lr=2.0):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)
        # self.init_lr = lr

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Utils function')
    parser.add_argument('--mode', type=str, default='vocab', 
            help='how to run the utils.py, (vocab,)')
    parser.add_argument('--dataset', type=str, default='dailydialog')
    parser.add_argument('--file', type=str, nargs='+', default=None, 
            help='file for generating the vocab')
    parser.add_argument('--vocab', type=str, default='',
            help='input or output vocabulary')
    parser.add_argument('--cutoff', type=int, default=0,
            help='cutoff of the vocabulary')
    parser.add_argument('--pretrained', type=str, default=None,
            help='Pretrained embedding file')
    parser.add_argument('--graph', type=str, default='./processed/dailydialog/train-graph.pkl')
    parser.add_argument('--maxlen', type=int, default=50)
    parser.add_argument('--src_vocab', type=str, default='./processed/zh50/iptvocab.pkl')
    parser.add_argument('--tgt_vocab', type=str, default='./processed/zh50/optvocab.pkl')
    parser.add_argument('--src', type=str, default='./data/dailydialog/src-train.pkl')
    parser.add_argument('--tgt', type=str, default='./data/dailydialog/tgt-train.pkl')
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--bidir', dest='bidir', action='store_true')
    parser.add_argument('--no-bidir', dest='bidir', action='store_false')
    parser.add_argument('--hops', type=int, default=3)
    parser.add_argument('--perturbation_in', type=str, default=None)
    parser.add_argument('--perturbation_out', type=str, default=None)
    parser.add_argument('--perturbation_mode', type=int, default=1)
    parser.add_argument('--ngram', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lang', type=str, default='en')
    parser.add_argument('--ctx', type=int, default=200)
    parser.add_argument('--fully', dest='fully', action='store_true')
    parser.add_argument('--no-fully', dest='fully', action='store_false')
    parser.add_argument('--self-loop', dest='self_loop', action='store_true')
    parser.add_argument('--no-self-loop', dest='self_loop', action='store_false')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--tgt_maxlen', type=int, default=30)
    
    args = parser.parse_args()

    mode = args.mode
    if mode == 'vocab':
        print(args.file)
        generate_vocab(args.file, args.vocab, cutoff=args.cutoff)
        # analyse_coverage_word_embedding(args.vocab, lang=args.lang)
    elif mode == 'pretrained':
        with open(args.vocab, 'rb') as f:
            vocab = pickle.load(f)
        generate_bert_embedding(vocab, args.pretrained)
    elif mode == 'graph':
        # save the preprocessed data for generating graph
        src_dataset, src_user, tgt_dataset, tgt_user = load_data(args.src, args.tgt, args.src_vocab, args.tgt_vocab, args.maxlen, args.tgt_maxlen)
        print(f'[!] prepare for preprocessing')
        ppdataset = idx2sent(src_dataset, args.src_vocab)
        print(f'[!] begin to create the graph')
        # ipdb.set_trace()
        generate_graph(ppdataset, args.graph, threshold=args.threshold,
                       bidir=args.bidir, lang=args.lang, fully=args.fully,
                       self_loop=args.self_loop)
    elif mode == 'stat':
        try:
            analyse_graph(f'./processed/{args.dataset}/{args.split}-graph.pkl',
                          hops=args.hops)
        except:
            pass
        analyse_dataset(args.dataset, args.split)
    elif mode == 'perturbation':
        if args.perturbation_in and args.perturbation_out:
            Perturbations_test(args.perturbation_in, args.perturbation_out, mode=args.perturbation_mode)
        else:
            print(f'[!] check the perturbation file path')
    elif mode == 'lm':
        data = read_file(f'./data/{args.dataset}/src-train.txt')
        train_ngram_lm(args.dataset, data, ngram=args.ngram, gamma=args.gamma)
    elif mode == 'preprocess_transformer':
        src_train_path = f'data/{args.dataset}/src-train.txt'
        tgt_train_path = f'data/{args.dataset}/tgt-train.txt'
        train_path = f'data/{args.dataset}/train.txt'
        src_test_path = f'data/{args.dataset}/src-test.txt'
        tgt_test_path = f'data/{args.dataset}/tgt-test.txt'
        test_path = f'data/{args.dataset}/test.txt'
        src_dev_path = f'data/{args.dataset}/src-dev.txt'
        tgt_dev_path = f'data/{args.dataset}/tgt-dev.txt'
        dev_path = f'data/{args.dataset}/dev.txt'
        transformer_preprocess(src_train_path, tgt_train_path, train_path, ctx=args.ctx)
        transformer_preprocess(src_test_path, tgt_test_path, test_path, ctx=args.ctx)
        transformer_preprocess(src_dev_path, tgt_dev_path, dev_path, ctx=args.ctx)
    else:
        print(f'[!] wrong mode to run the script')
