#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.9.15
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler
import torch.optim as optim
import random
import numpy as np
import argparse
import math
import pickle
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import ipdb
import transformers
import gensim
from shutil import copyfile
from utils import *
from data_loader import *
from metric.metric import *
from model.PHAED import PHAED
from utils import PAD_ID
import context_cache as ctx
from torch.cuda.amp import autocast, GradScaler 

scaler = GradScaler()
def train(train_iter, model, optimizer, vocab_size, pad,
          grad_clip=10, debug=False):
    # choose nll_loss for training the objective function
    model.train()
    total_loss, total_nll_loss, batch_num = 0.0, 0.0, 0
    # criterion = nn.NLLLoss(ignore_index=pad, reduction='none')
    criterion = nn.NLLLoss(ignore_index=pad, reduction='none')
    out_of_memory_count = 0
    pbar = tqdm(train_iter)
    truncated_turn = 4
    nll_loss_item = 0.0
    step_count = 0
    optimizer.zero_grad()
    freq = 1
    for idx, batch in enumerate(pbar):
        # [batch, turn, length], [batch, seq_len, batch] / [batch, seq_len], [batch, seq_len]
        src, tbatch, lengths = batch
        src_turn_lengths, tgt_turn_lengths, \
        sbatch_user, tbatch_user, \
        sbatch_turn_id, tbatch_turn_id = lengths
        
        sbatch, sents_mapping, sent_rank = src
        batch_size = sbatch.shape[0]
        if batch_size == 1:
            # batchnorm will throw error when batch_size is 1
            continue
        n_sents = 0
        enc_out = 0
        nll_loss = 0 
        loss = 0.0
        words_norm = 0
        try:
            with autocast():
                enc_out, enc_mask = model.encoder(sbatch, position=sent_rank,
                                                  segment_ids=sents_mapping, turn_inp=sbatch_turn_id)
                ctx.memory_cache = tuple()
                ctx.memory_mask = None
                ctx.memory_pad_num = None
                y_batch = tbatch # torch.einsum('btl->tbl', tbatch)
                n_sents = len(y_batch)
                for sents_no, y_sents in enumerate(y_batch):
                    y_inp = y_sents[:, :-1].contiguous()
                    y_label = y_sents[:, 1:].contiguous()
                    ## y_label = torch.cat([y_sents[:, 1:].contiguous().clone(), y_sents.new_zeros(y_sents.size(0),1).fill_(PAD_ID)], dim=1).contiguous()
                    is_not_current_sents = sents_mapping.detach().ne(sents_no)
                    current_sent_mask = torch.where(is_not_current_sents, is_not_current_sents, enc_mask)
                    y_turn_id = tbatch_turn_id[:, sents_no]
                    log_probs = model.decode_train(y_inp, y_turn_id, enc_out, current_sent_mask, log_probs=True)
                    new_mask = y_label.eq(PAD_ID).view(-1, y_label.size(-1)).transpose(0, 1)
                    ctx.memory_mask = new_mask if ctx.memory_mask is None else torch.cat([ctx.memory_mask, new_mask], dim=0)
                    memory_pad_num = new_mask.new_zeros(new_mask.size()) + new_mask.sum(0)[None, :]
                    ctx.memory_pad_num = memory_pad_num if ctx.memory_pad_num is None else torch.cat([ctx.memory_pad_num+new_mask.sum(0)[None, :], memory_pad_num], dim=0)
    
                    if ctx.knowlege_length<=sents_no and sents_no <= n_sents - 1:
                        # loss_ = criterion(log_probs.view(-1, log_probs.size(-1)), y_label.view(-1)).view((batch_size, -1)).sum(-1)
                        words_norm += y_label.ne(pad).float().sum()
                        # loss += loss_.div(words_norm).sum()
                        tmp_loss = criterion(log_probs.view(-1, log_probs.size(-1)), y_label.view(-1))
                        tmp_loss = tmp_loss.view(batch_size, y_label.size(1))
                        loss += (tmp_loss.sum(1)).mean(0)
                    assert sents_no <= n_sents - 1
                
            batch_num += 1
            #loss.backward()
             
            
            nll_loss_token = loss.item() * batch_size / words_norm
            nll_loss_item = nll_loss_token.item()
            #loss.backward()
            loss = loss / freq
            scaler.scale(loss).backward()

            if batch_num % freq == 0:
                clip_grad_norm_(model.parameters(), grad_clip)
                # optimizer.step()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += nll_loss_item
            out_of_memory_count = 0
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: out of memory, n_sents:", n_sents)
                enc_out = 0
                enc_mask = None
                nll_loss = 0
                loss = 0
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                out_of_memory_count += 1
                optimizer.zero_grad()
                if out_of_memory_count>20:
                    raise exception
                continue
            else:
                raise exception
        
        pbar.set_description(
            f'batch {batch_num}, nll loss: {round(nll_loss_item, 4)}, training loss: {round(total_loss/batch_num, 4)}')
        
    return round(total_loss / batch_num, 4)
    

def validation(data_iter, model, vocab_size, pad, debug=False):
    model.eval()
    total_loss, batch_num, total_word_num = 0.0, 0, 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=pad, reduction='sum')

    pbar = tqdm(data_iter)
    
    for idx, batch in enumerate(pbar):
        # [batch, turn, length], [batch, seq_len, batch] / [batch, seq_len], [batch, seq_len]
        src, tbatch, lengths = batch
        src_turn_lengths, tgt_turn_lengths, \
        sbatch_user, tbatch_user, \
        sbatch_turn_id, tbatch_turn_id = lengths
        
        sbatch, sents_mapping, sent_rank = src
        batch_size = sbatch.shape[0]
        if batch_size == 1:
            # batchnorm will throw error when batch_size is 1
            continue
        nll_loss = 0

        enc_out, enc_mask = model.encoder(sbatch, position=sent_rank,
                                          segment_ids=sents_mapping, turn_inp=sbatch_turn_id)
        ctx.memory_cache = tuple()
        ctx.memory_mask = None
        ctx.memory_pad_num = None
        sum_n = 0.0
        
        y_batch = tbatch # torch.einsum('btl->tbl', tbatch)
        n_sents = len(y_batch)
        for sents_no, y_sents in enumerate(y_batch):
            y_inp = y_sents[:, :-1].contiguous()
            y_label = y_sents[:, 1:].contiguous()
            is_not_current_sents = sents_mapping.detach().ne(sents_no)
            current_sent_mask = torch.where(is_not_current_sents, is_not_current_sents, enc_mask)
            y_turn_id = tbatch_turn_id[:, sents_no]
            log_probs = model.decode_train(y_inp, y_turn_id, enc_out, current_sent_mask, log_probs=True)
            new_mask = y_label.eq(PAD_ID).view(-1, y_label.size(-1)).transpose(0, 1)
            ctx.memory_mask = new_mask if ctx.memory_mask is None else torch.cat([ctx.memory_mask, new_mask], dim=0)
            memory_pad_num = new_mask.new_zeros(new_mask.size()) + new_mask.sum(0)[None, :]
            ctx.memory_pad_num = memory_pad_num if ctx.memory_pad_num is None else torch.cat([ctx.memory_pad_num+new_mask.sum(0)[None, :], memory_pad_num], dim=0)
            assert sents_no <= n_sents - 1
            if sents_no == n_sents - 1:
                assert log_probs.size(0) == y_label.size(0)
                log_probs = log_probs[:, 1:].contiguous()
                y_label = y_label[:, 1:].contiguous()
                nll_loss = criterion(log_probs.view(-1, vocab_size), y_label.view(-1))
                non_pad_mask = y_label.ne(pad).long()
                word_num = non_pad_mask.sum().item()
                assert word_num>0
                total_word_num += word_num
                model.finish_decoder()
            ctx.IS_INFERRING = False
        total_loss += nll_loss.item()
        assert nll_loss != 0
        batch_num += 1
        pbar.set_description(f'batch {batch_num}, nll loss: {round(nll_loss.item()/word_num, 4)}, {total_word_num}')
    
    return round(total_loss / total_word_num, 4)


def translate(data_iter, net, epoch=0, **kwargs):
    '''
    PPL calculating refer to: https://github.com/hsgodhia/hred
    '''
    net.eval()
    tgt_vocab = load_pickle(kwargs['tgt_vocab'])
    src_vocab = load_pickle(kwargs['src_vocab'])
    src_w2idx, src_idx2w = src_vocab
    tgt_w2idx, tgt_idx2w = tgt_vocab
    # calculate the loss
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_w2idx['<pad>'], reduction='sum')
    total_loss, batch_num, total_word_num = 0.0, 0, 0.0

    # translate, which is the same as the translate.py
    
    pred_path = kwargs['pred'][:-4]+str(epoch)+'.txt'
    with open(pred_path, 'w', encoding='utf-8') as f:
        pbar = tqdm(data_iter)
        for batch in pbar:
            src, tbatch, lengths = batch
            src_turn_lengths, tgt_turn_lengths, \
            sbatch_user, tbatch_user, \
            sbatch_turn_id, tbatch_turn_id = lengths

            sbatch, sents_mapping, sent_rank = src
            batch_size = sbatch.shape[0]
            turn_size = len(sbatch[0])

            src_pad, tgt_pad = src_w2idx['<pad>'], tgt_w2idx['<pad>']
            src_eos, tgt_eos = src_w2idx['<eos>'], tgt_w2idx['<eos>']
            vocab_size = len(tgt_w2idx)

            # output: [maxlen, batch_size], sbatch: [turn, max_len, batch_size]
            with torch.no_grad():
                ctx.memory_cache = tuple()
                ctx.memory_mask = None
                output, _ = net.predict(src, sbatch_turn_id, tbatch[:-1], tbatch_turn_id[:, :-1], maxlen=kwargs['tgt_maxlen'], loss=True)
            # true working ppl by using teach_force
            with torch.no_grad():
                enc_out, enc_mask = net.encoder(sbatch, position=sent_rank,
                                                segment_ids=sents_mapping, turn_inp=sbatch_turn_id)
                ctx.memory_cache = tuple()
                ctx.memory_mask = None
                ctx.memory_pad_num = None
                nll_loss_item = 0
                
                y_batch = tbatch
                n_sents = len(y_batch)
                for sents_no, y_sents in enumerate(y_batch):
                    y_inp = y_sents[:, :-1].contiguous()
                    y_label = y_sents[:, 1:].contiguous()
                    is_not_current_sents = sents_mapping.detach().ne(sents_no)
                    current_sent_mask = torch.where(is_not_current_sents, is_not_current_sents, enc_mask)
                    y_turn_id = tbatch_turn_id[:, sents_no]
                    
                    log_probs = net.decode_train(y_inp, y_turn_id, enc_out, current_sent_mask, log_probs=True)
                    new_mask = y_label.eq(PAD_ID).view(-1, y_label.size(-1)).transpose(0, 1)
                    ctx.memory_mask = new_mask if ctx.memory_mask is None else torch.cat([ctx.memory_mask, new_mask], dim=0)
                    memory_pad_num = new_mask.new_zeros(new_mask.size()) + new_mask.sum(0)[None, :]
                    ctx.memory_pad_num = memory_pad_num if ctx.memory_pad_num is None else torch.cat([ctx.memory_pad_num+new_mask.sum(0)[None, :], memory_pad_num], dim=0)
                    if sents_no == n_sents - 1:
                        assert log_probs.size(0) == y_label.size(0)
                        log_probs = log_probs[:, 1:].contiguous()
                        y_label = y_label[:, 1:].contiguous()
                        nll_loss = criterion(log_probs.view(-1, vocab_size), y_label.view(-1))
                        nll_loss_item = nll_loss.item()
                        non_pad_mask = y_label.ne(tgt_pad).long()
                        word_num = non_pad_mask.sum().item()
                        total_word_num += word_num
                        net.finish_decoder()
                    assert sents_no <= n_sents - 1
            # teach_force over
            assert nll_loss_item != 0
            batch_num += 1
            total_loss += nll_loss_item 

            # ipdb.set_trace()
            for i in range(batch_size):
                ref = list(map(int, tbatch[-1][i, :].tolist()))
                tgt = list(map(int, output[:, i].tolist()))  # [maxlen]
                src = list()
                if kwargs['hierarchical']:
                    began = 0
                    j = 0
                    for k, id in enumerate(sbatch[i]):
                        if id == EOS_ID:
                            src += ([sbatch[i, began:k+1].tolist()] + [tbatch[j][i].tolist()]) # [turns, maxlen]
                            began = k+1
                            j += 1
                else:
                    src = list(map(int, sbatch[:, i].tolist()))
                # filte the <pad>
                ref_endx = ref.index(tgt_pad) if tgt_pad in ref else len(ref)
                ref_endx_ = ref.index(tgt_eos) if tgt_eos in ref else len(ref)
                ref_endx = min(ref_endx, ref_endx_)
                ref = ref[:ref_endx]
                ref = ' '.join(num2seq(ref, tgt_idx2w))
                ref = ref.replace('<sos>', '').strip()
                ref = ref.replace('<pad_utterance>', '').strip()
                ref = ref.replace('< user1 >', '').strip()
                ref = ref.replace('< user0 >', '').strip()

                tgt_endx = tgt.index(tgt_pad) if tgt_pad in tgt else len(tgt)
                tgt_endx_ = tgt.index(tgt_eos) if tgt_eos in tgt else len(tgt)
                tgt_endx = min(tgt_endx, tgt_endx_)
                tgt_id = tgt
                tgt = tgt[:tgt_endx]
                tgt = ' '.join(num2seq(tgt, tgt_idx2w))
                tgt_old = tgt
                tgt = tgt.replace('<sos>', '').strip()
                tgt = tgt.replace('< user1 >', '').strip()
                tgt = tgt.replace('< user0 >', '').strip()
                if len(tgt) < 1 or tgt_endx<2:
                    print('\nempty tgt_old:', tgt_old)
                    print('empty tgt_id:', tgt_id)
                    print('empty tgt_id2wd:', ' '.join(num2seq(tgt_id, tgt_idx2w)))
                    print('empty tgt:', tgt)

                if kwargs['hierarchical']:
                    source = []
                    for item in src[:-1]:
                        item_endx = item.index(src_pad) if src_pad in item else len(item)
                        item_endx_ = item.index(src_eos) if src_eos in item else len(item)
                        item_endx = min(item_endx, item_endx_)
                        item = item[1:item_endx]
                        item = num2seq(item, src_idx2w)
                        tmp = ' '.join(item).replace('<pad_utterance>', '').strip()
                        if tmp != '':
                            source.append(tmp)
                    src = ' __eou__ '.join(source)
                else:
                    src_endx = src.index(src_pad) if src_pad in src else len(src)
                    src_endx_ = src.index(src_eos) if src_eos in src else len(src)
                    src_endx = min(src_endx, src_endx_)
                    src = src[1:src_endx]
                    src = ' '.join(num2seq(src, src_idx2w))
                # print('--src:', src)
                # print('--ref:', ref)
                # print('--tgt:', tgt)
                f.write(f'- src: {src}\n')
                f.write(f'- ref: {ref}\n')
                f.write(f'- tgt: {tgt}\n\n')
    copyfile(pred_path, kwargs['pred'])
    l = round(total_loss / total_word_num, 4)
    print(f'[!] write the translate result into {kwargs["pred"]}')
    print(f'[!] epoch: {epoch}, test loss: {l}, test ppl: {round(math.exp(l), 4)}',
          file=open(f'./processed/{kwargs["dataset"]}/{kwargs["model"]}/trainlog.txt', 'a'))

    return math.exp(l)


def write_into_tb(pred_path, writer, writer_str, epoch, ppl, model, dataset):
    # obtain the performance
    print(f'[!] measure the performance and write into tensorboard')
    with open(pred_path, encoding='utf-8') as f:
        ref, tgt = [], []
        for idx, line in enumerate(f.readlines()):
            line = line.lower()  # lower the case
            if idx % 4 == 1:
                line = line.replace("user1", "").replace("user0", "").replace("- ref: ", "").replace('<sos>',
                                                                                                     '').replace(
                    '<eos>', '').strip()
                ref.append(line.split())
            elif idx % 4 == 2:
                line = line.replace("user1", "").replace("user0", "").replace("- tgt: ", "").replace('<sos>',
                                                                                                     '').replace(
                    '<eos>', '').strip()
                tgt.append(line.split())

    assert len(ref) == len(tgt)

    # ROUGE
    rouge_sum, bleu1_sum, bleu2_sum, bleu3_sum, bleu4_sum, counter = 0, 0, 0, 0, 0, 0
    for rr, cc in tqdm(list(zip(ref, tgt))):
        rouge_sum += cal_ROUGE(rr, cc)
        # rouge_sum += 0.01
        counter += 1

    # BlEU
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
    rintra_dist1, rintra_dist2, rinter_dist1, rinter_dist2 = distinct(ref)

    # bert_scores = cal_BERTScore(refs, tgts)

    # Embedding-based metric: Embedding Average (EA), Vector Extrema (VX), Greedy Matching (GM)
    # load the dict
    # with open('./data/glove_embedding.pkl', 'rb') as f:
    #     dic = pickle.load(f)
    dic = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
    print('[!] load the GoogleNews 300 word2vector by gensim over')
    ea_sum, vx_sum, gm_sum, counterp = 0, 0, 0, 0
    for rr, cc in tqdm(list(zip(ref, tgt))):
        ea_sum += cal_embedding_average(rr, cc, dic)
        vx_sum += cal_vector_extrema(rr, cc, dic)
        gm_sum += cal_greedy_matching_matrix(rr, cc, dic)
        counterp += 1
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

    # write into the tensorboard
    writer.add_scalar(f'{writer_str}-Performance/PPL', ppl, epoch)
    writer.add_scalar(f'{writer_str}-Performance/BLEU-1', bleu1_sum, epoch)
    writer.add_scalar(f'{writer_str}-Performance/BLEU-2', bleu2_sum, epoch)
    writer.add_scalar(f'{writer_str}-Performance/BLEU-3', bleu3_sum, epoch)
    writer.add_scalar(f'{writer_str}-Performance/BLEU-4', bleu4_sum, epoch)

    writer.add_scalar(f'{writer_str}-Performance/Distinct-1', distinct_1, epoch)
    writer.add_scalar(f'{writer_str}-Performance/Distinct-2', distinct_2, epoch)
    writer.add_scalar(f'{writer_str}-Performance/Ref-Distinct-1', rdistinct_1, epoch)
    writer.add_scalar(f'{writer_str}-Performance/Ref-Distinct-2', rdistinct_2, epoch)

    writer.add_scalar(f'{writer_str}-Performance/Intra-Distinct-1', intra_dist1, epoch)
    writer.add_scalar(f'{writer_str}-Performance/Intra-Distinct-2', intra_dist2, epoch)
    writer.add_scalar(f'{writer_str}-Performance/Inter-Distinct-1', inter_dist1, epoch)
    writer.add_scalar(f'{writer_str}-Performance/Inter-Distinct-2', inter_dist2, epoch)


    writer.add_scalar(f'{writer_str}-Performance/Metric_Average', np.mean(metric_average), epoch)
    writer.add_scalar(f'{writer_str}-Performance/Metric_Extrema', np.mean(metric_extrema), epoch)
    writer.add_scalar(f'{writer_str}-Performance/Metric_Greedy', np.mean(metric_greedy), epoch)

    writer.add_scalar(f'{writer_str}-Performance/Length-Average-tgt', np.mean(length_tgt), epoch)
    writer.add_scalar(f'{writer_str}-Performance/Length-Average-ref', np.mean(length_ref), epoch)
    
    
    print('n_sent', n_sent)


    # write now
    writer.flush()


def main(**kwargs):
    # tensorboard
    writer = SummaryWriter(log_dir=f'./tblogs/{kwargs["dataset"]}/{kwargs["model"]}')

    # load vocab
    src_vocab, tgt_vocab = load_pickle(kwargs['src_vocab']), load_pickle(kwargs['tgt_vocab'])
    src_w2idx, src_idx2w = src_vocab
    tgt_w2idx, tgt_idx2w = tgt_vocab
    print(f'[!] load vocab over, src/tgt vocab size: {len(src_idx2w)}, {len(tgt_w2idx)}')

    # pretrained path
    pretrained = None
    ctx.knowlege_length = kwargs['knowlege_length']
    print('ctx.knowlege_length:', ctx.knowlege_length)

    # create the net
    net = PHAED(len(src_w2idx), len(tgt_w2idx),
                    kwargs['num_encoder_layers'], kwargs['num_decoder_layers'], kwargs['n_head'], kwargs['embed_size'], kwargs['d_model'], kwargs['dim_feedforward'],
                    num_pos_embeddings=kwargs['position_embed_size'], num_turn_embeddings=kwargs['num_turn_embeddings'],
                    padding_idx=tgt_w2idx['<pad>'], sos_idx=tgt_w2idx['<sos>'],
                    dropout=kwargs['dropout'], teach_force=kwargs['teach_force'],
                    proj_share_weight = 'Ubuntu' in kwargs['src_train']
                    )

    if torch.cuda.is_available():
        net.cuda()

    print('[!] Net:')
    print(net)

    print(f'[!] Parameters size: {sum(x.numel() for x in net.parameters())}')
    print(f'[!] Optimizer Adam')
    optimizer = optim.Adam(net.parameters(), lr=kwargs['lr'])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                               mode='min',
                                               factor=kwargs['lr_gamma'],
                                               patience=kwargs['patience'],
                                               verbose=True,
                                               cooldown=0,
                                               min_lr=kwargs['lr_mini'])
     
    pbar = tqdm(range(1, kwargs['epochs'] + 1))
    training_loss, validation_loss = [], []
    min_loss = np.inf
    patience = 0
    best_val_loss = None
    teacher_force_ratio = kwargs['teach_force']  # default 1
    # holder = teacher_force_ratio_counter
    holder = 0 
    best_checkpoint_path = ''
    checkpoint_path = kwargs['checkpoint_path'] #'ckpt/Ubuntu/DHAED_6layer/vloss_3.4671_epoch_12_best.pt'
    # train
    if os.path.exists(checkpoint_path) and 'Ubuntu' in kwargs['src_train']:
        state = torch.load(checkpoint_path)
        net.load_state_dict(state['net'])
        epoch = state['epoch']
        patience = state['patience']
        optimizer.load_state_dict(state['opt']) # optimizer = optim.Adam(net.parameters(), lr=kwargs['lr'])
        pbar = tqdm(range(epoch+1, kwargs['epochs'] + 1))
        print('load model stat:', checkpoint_path)
        
    
    # train
    for epoch in pbar:
        # prepare dataset
        train_iter = get_batch_data_hier_tf_random(kwargs['src_train'], kwargs['tgt_train'], kwargs['src_vocab'],
                                            kwargs['tgt_vocab'], kwargs['batch_size'], kwargs['maxlen'],
                                            kwargs['tgt_maxlen'], ld=True)
        test_iter = get_batch_data_hier_tf(kwargs['src_test'], kwargs['tgt_test'], kwargs['src_vocab'],
                                           kwargs['tgt_vocab'], kwargs['batch_size']*4, kwargs['maxlen'],
                                           kwargs['tgt_maxlen'], ld=True, random=True)
        dev_iter = get_batch_data_hier_tf(kwargs['src_dev'], kwargs['tgt_dev'], kwargs['src_vocab'],
                                          kwargs['tgt_vocab'], kwargs['batch_size']*2, kwargs['maxlen'],
                                          kwargs['tgt_maxlen'], ld=True, random=True)

        print(f'Epoch {epoch}:')
 
         

        # ========== train session begin ==========
        writer_str = f'{kwargs["dataset"]}'
        train_loss = train(train_iter, net, optimizer, len(tgt_w2idx),
                           tgt_w2idx['<pad>'],
                           grad_clip=kwargs['grad_clip'], debug=kwargs['debug'], ) 
        #if epoch<=10:
        #    continue
        with torch.no_grad():
            val_loss = validation(dev_iter, net, len(tgt_w2idx),
                                  tgt_w2idx['<pad>'])
        
        # add loss scalar to tensorboard
        # and write the lr schedule, and teach force
        writer.add_scalar(f'{writer_str}-Loss/train', train_loss, epoch)
        writer.add_scalar(f'{writer_str}-Loss/dev', val_loss, epoch)
        writer.add_scalar(f'{writer_str}-Loss/lr',
                          optimizer.state_dict()['param_groups'][0]['lr'],
                          epoch)
        writer.add_scalar(f'{writer_str}-Loss/teach', net.teach_force,
                          epoch)
        scheduler.step(val_loss)

        if not best_val_loss or val_loss < best_val_loss or epoch>10:
            best_val_loss = val_loss
            patience = 0
            optim_state = optimizer.state_dict()
            state = {'net': net.state_dict(), 'opt': optim_state,
                     'epoch': epoch, 'patience': patience}
            torch.save(state,
                       f'./ckpt/{kwargs["dataset"]}/{kwargs["model"]}/vloss_{val_loss}_epoch_{epoch}_best.pt')
            if os.path.exists(best_checkpoint_path) and epoch <= 10:
                os.remove(best_checkpoint_path)
            best_checkpoint_path = f'./ckpt/{kwargs["dataset"]}/{kwargs["model"]}/vloss_{val_loss}_epoch_{epoch}_best.pt'
        else:
            patience += 1
        
        #if math.exp(val_loss)>=30.0:
        #    continue 

        # translate on test dataset
        with torch.no_grad():
            ppl = translate(test_iter, net, epoch=epoch, **kwargs)
        # write the performance into the tensorboard
        write_into_tb(kwargs['pred'], writer, writer_str, epoch, ppl, kwargs['model'], kwargs['dataset'])

        pbar.set_description(
            f'Epoch: {epoch}, tfr: {round(teacher_force_ratio, 4)}, loss(train/dev): {train_loss}/{val_loss}, ppl(dev/test): {round(math.exp(val_loss), 4)}/{round(ppl, 4)}, patience: {patience}/{kwargs["patience"]}')

        

        # lr schedule change, monitor the evaluation loss

        if patience > kwargs["patience"]:
            print(f'patience:{patience} > kwargs["patience"]:{kwargs["patience"]}')
            break
    pbar.close()
    writer.close()
    print(f'[!] Done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('--src_train', type=str, default=None, help='src train file')
    parser.add_argument('--tgt_train', type=str, default=None, help='src train file')
    parser.add_argument('--src_test', type=str, default=None, help='src test file')
    parser.add_argument('--tgt_test', type=str, default=None, help='tgt test file')
    parser.add_argument('--src_dev', type=str, default=None, help='src dev file')
    parser.add_argument('--tgt_dev', type=str, default=None, help='tgt dev file')
    parser.add_argument('--min_threshold', type=int, default=0, 
                        help='epoch threshold for loading best model')
    parser.add_argument('--max_threshold', type=int, default=20, 
                        help='epoch threshold for loading best model')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--model', type=str, default='HRED', help='model to be trained')
    parser.add_argument('--teach_force', type=float, default=1.0, 
                        help='teach force ratio')
    parser.add_argument('--seed', type=int, default=30,
                        help='random seed')
    parser.add_argument('--embed_size', type=int, default=200, 
                        help='embedding layer size')
    parser.add_argument('--patience', type=int, default=5, 
                        help='patience for early stop')
    parser.add_argument('--dataset', type=str, default='dailydialog', 
                        help='dataset for training')
    parser.add_argument('--grad_clip', type=float, default=10.0, help='grad clip')
    parser.add_argument('--epochs', type=int, default=20, help='epochs for training')
    parser.add_argument('--src_vocab', type=str, default=None, help='src vocabulary')
    parser.add_argument('--tgt_vocab', type=str, default=None, help='tgt vocabulary')
    parser.add_argument('--maxlen', type=int, default=50, help='the maxlen of the utterance')
    parser.add_argument('--tgt_maxlen', type=int, default=50, 
                        help='the maxlen of the responses')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--no-debug', dest='debug', action='store_false')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout ratio')
    parser.add_argument('--hierarchical', type=int, default=1, 
                        help='Whether hierarchical architecture')
    parser.add_argument('--d_model', type=int, default=512, 
                        help='d_model for transformer')
    parser.add_argument('--n_head', type=int, default=8, 
                        help='head number for transformer')
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--num_turn_embeddings', type=int, default=50)
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--position_embed_size', type=int, default=30)
    parser.add_argument('--knowlege_length', type=int, default=0, 
                        help='knowlege_length') 
    
    parser.add_argument('--pred', type=str, default=None, 
                        help='the file save the output')
    parser.add_argument('--checkpoint_path', type=str, default='', 
                        help='the checkpoint path')               
    parser.add_argument('--dynamic_tfr', type=int, default=20, 
                        help='begin to use the dynamic teacher forcing ratio, each ratio minus the tfr_weight')
    parser.add_argument('--lr_mini', type=float, default=1e-6, help='minial lr (threshold)')
    parser.add_argument('--lr_gamma', type=float, default=0.8, help='lr schedule gamma factor')

    args = parser.parse_args()
    #################################################################
    ##################################################################
    # show the parameters and write into file
    print('[!] Parameters:')
    print(args)
    with open(f'./processed/{args.dataset}/{args.model}/metadata.txt', 'w') as f:
        print(vars(args), file=f)
    # set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # main function
    args_dict = vars(args)
    main(**args_dict)
