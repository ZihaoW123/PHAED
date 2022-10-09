import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm
import random
import ipdb
import math

from utils import *
from data_loader import *
from metric.metric import *

from model.PHAED import PHAED
import context_cache as ctx


def translate(**kwargs): 
    src_vocab, tgt_vocab = load_pickle(kwargs['src_vocab']), load_pickle(kwargs['tgt_vocab'])
    src_w2idx, src_idx2w = src_vocab
    tgt_w2idx, tgt_idx2w = tgt_vocab
    # load dataset
    test_iter = get_batch_data_hier_tf(kwargs['src_test'], kwargs['tgt_test'], kwargs['src_vocab'],
                                           kwargs['tgt_vocab'], kwargs['batch_size'], kwargs['maxlen'],
                                           kwargs['tgt_maxlen'], ld=True, random=False)


    # pretrained mode
    pretrained = None
    
    # load net
    net = PHAED(len(src_w2idx), len(tgt_w2idx),
                    kwargs['num_encoder_layers'], kwargs['num_decoder_layers'], kwargs['n_head'], kwargs['embed_size'], kwargs['d_model'], kwargs['dim_feedforward'],
                    num_pos_embeddings=kwargs['position_embed_size'], num_turn_embeddings=kwargs['num_turn_embeddings'],
                    padding_idx=tgt_w2idx['<pad>'], sos_idx=tgt_w2idx['<sos>'],
                    dropout=kwargs['dropout'], teach_force=kwargs['teach_force'],
                    proj_share_weight = 'Ubuntu' in kwargs['src_train']
                    )
    # load best model
    load_best_model(kwargs['dataset'], kwargs['model'], 
                    net, min_threshold=kwargs['min_threshold'],
                    max_threshold=kwargs["max_threshold"]) 
    if torch.cuda.is_available():
        net.cuda()
        net.eval()
        
    
    print('Net:')
    print(net)
    print(f'[!] Parameters size: {sum(x.numel() for x in net.parameters())}')

    
    net.eval()
    tgt_vocab = load_pickle(kwargs['tgt_vocab'])
    src_vocab = load_pickle(kwargs['src_vocab'])
    src_w2idx, src_idx2w = src_vocab
    tgt_w2idx, tgt_idx2w = tgt_vocab
    # calculate the loss
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_w2idx['<pad>'], reduction='sum')
    total_loss, batch_num, total_word_num = 0.0, 0, 0.0
    
    
    # translate
    pred_path = kwargs['pred']
    with open(pred_path, 'w', encoding='utf-8') as f:
        pbar = tqdm(test_iter)
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
    print(f'[!] epoch: {epoch}, test loss: {l}, test ppl: {round(math.exp(l), 4)}')


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
    parser.add_argument('--train_graph', type=str, default=None, 
                        help='train graph data path')
    parser.add_argument('--test_graph', type=str, default=None, 
                        help='test graph data path')
    parser.add_argument('--dev_graph', type=str, default=None, 
                        help='dev graph data path')
    
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
    with torch.no_grad():
        translate(**args_dict)