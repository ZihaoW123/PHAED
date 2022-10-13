import torch
import numpy as np
import os
import sys
import jieba
import re
import tqdm
import pickle
from string import punctuation as Einglishgpunction
from zhon.hanzi import punctuation as Chinesepunction
from collections import Counter
from torchtext import vocab
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pack_sequence, pad_packed_sequence
from models.model_utils.utils import gVar, include_chinese

# ========== For Zh ========== #
SPECIAL_TOKENS = ['[PAD]', '[SOUS1]', '[SOUS2]', '[EOU]', '[UNK]']
PAD, SOUS1, SOUS2, EOU, UNK = 0, 1, 2, 3, 4

path = os.path.join(sys.path[0].replace('data', ''), 'data')


# ========== For .txt ========== #
def read_text_data(path):
    new_data = []
    new_data_vocab = []
    with open(path, 'r', encoding='utf-8') as f:
        num = 0
        while True:
            data = f.readline()

            if not data:
                print(f'数据读取完成, path={path}, num={num}')
                break
            data = data.lower().split(' __eou__ ')

            new_data_ = []
            for ii in data:
                assert ii != '' and ii != '\n'
                new_data_.append(ii.strip())

            if len(new_data_) == 1:  ## For non-conversational text
                new_data.append(new_data_)
            else:
                new_data_vocab.append(new_data_)
                for ii in range(2, len(new_data_) + 1):
                    tmp = new_data_[:ii]
                    tmp = tmp[-10:]
                    new_data.append(tmp)
            num += 1
            if num % 100000 == 0:
                print(f'dialogue num={num}')
                # break
        print('num pairs', len(new_data))

    return new_data, new_data_vocab


def cut_clean_sentence(tgt, tgt_maxlen, tgt_w2idx):
    line = tgt[:]
    if len(line) > tgt_maxlen:
        # line = line[:tgt_maxlen-1] + [tgt_w2idx['<eos>']]
        line = line[:tgt_maxlen - 1]
        len_line = len(line)
        line_re = line[:]
        line_re.reverse()
        end_1 = len_line - line_re.index(tgt_w2idx['。']) if tgt_w2idx['。'] in line_re else -1
        end_2 = len_line - line_re.index(tgt_w2idx['？']) if tgt_w2idx['？'] in line_re else -1
        end_3 = len_line - line_re.index(tgt_w2idx['！']) if tgt_w2idx['！'] in line_re else -1
        end = max(end_1, end_2, end_3)
        if end == -1:
            end = len_line - line_re.index(tgt_w2idx[',']) if tgt_w2idx[','] in line_re else -1
            if end == -1:
                end = tgt_maxlen - 1
                line = line[:end] + [tgt_w2idx['[EOU]']]
            else:
                end = tgt_maxlen - 1
                line = line[:end - 1] + [tgt_w2idx['。'], tgt_w2idx['[EOU]']]
            assert len(line) <= tgt_maxlen
        else:
            line = line[:end] + [tgt_w2idx['[EOU]']]
            assert len(line) <= tgt_maxlen
    return line


class ChineseTokenizer(object):
    '''
    Only for Chinese RNN based model, parameters:
    :corpus: a list of pair (context string, response string)
    '''

    def __init__(self, corpus, n_vocab=50000, min_freq=4, spliter=' ', char_seg=True):
        self.allowPOS = ['n', 'nr', 'nz', 'PER', 'LOC', 'ORG', 'ns', 'nt', 'nw', 'vn', 's']
        self.topk = 10
        self.spliter = spliter
        self.char_seg = char_seg
        special_tokens = SPECIAL_TOKENS
        self.vocab = vocab.Vocab(
            self._build_vocab(corpus),
            max_size=n_vocab,
            min_freq=min_freq,
            specials=special_tokens,
        )
        assert self.vocab.stoi['[PAD]'] == 0, f'[PAD] id should be 0, but got {self.vocab.stoi["[PAD]"]}'
        print(f'[!] init the vocabulary over, vocab size: {len(self.vocab)}')
        print(f'[!] init the vocabulary over, vocab: {self.vocab.itos[:10000]}')

    def __len__(self):
        return len(self.vocab)

    def size(self):
        return len(self.vocab)

    def decode(self, idx_seq, spliter=' '):
        words = self.idx2toks(idx_seq)
        return spliter.join(words)

    def encode(self, tok_seq, len_per_dialogue=None, len_per_utterance=None, num=1):
        '''Careful about the special tokens'''
        sentences = re.split('\[EOU\]', tok_seq)
        eou_token = self.vocab.stoi['[EOU]']
        sou_token = [self.vocab.stoi['[SOUS1]'], self.vocab.stoi['[SOUS2]']]
        idxs = []
        for sentence in sentences:
            if sentence == '':
                return []
            sentence = sentence.strip()
            if not self.char_seg:
                sentence = list(jieba.cut(sentence))
            else:
                if self.spliter is None:
                    sentence_ = list(jieba.cut(sentence))
                else:
                    sentence_ = sentence.split(self.spliter)
                sentence = []
                for tmp in sentence_:
                    if include_chinese(tmp):
                        sentence += list(tmp)
                    else:
                        sentence.append(tmp.lower())
            sentence = list(
                map(lambda i: self.vocab.stoi[i] if i in self.vocab.stoi else self.vocab.stoi['[UNK]'], sentence))

            sentence = [sou_token[num % 2]] + sentence + [eou_token]
            if (len_per_utterance is not None) and len(sentence) > len_per_utterance:
                sentence = cut_clean_sentence(sentence, len_per_utterance, self.vocab.stoi)
            idxs.extend(sentence)
            num -= 1

        if (len_per_dialogue is not None) and (len(idxs) > len_per_dialogue):
            idxs_ = idxs[:-(len_per_dialogue - 1)]
            idxs_.reverse()
            s1, s2 = 0, 0
            if idxs_.count(sou_token[0]) > 0:
                s1 = len(idxs_) - 1 - idxs_.index(sou_token[0])
            if idxs_.count(sou_token[1]) > 0:
                s2 = len(idxs_) - 1 - idxs_.index(sou_token[1])
            s = max(s1, s2)

            if s >= 0:
                assert idxs[s] == sou_token[1] or idxs[s] == sou_token[0]
                idxs = [idxs[s]] + idxs[-(len_per_dialogue - 1):]
            else:
                assert idxs[0] == sou_token[1] or idxs[0] == sou_token[0]
                idxs = idxs[:len_per_dialogue]
        return idxs

    def idx2toks(self, idx_seq):
        return list(map(lambda i: self.vocab.itos[i], idx_seq))

    def _build_vocab(self, corpus):
        vocab_counter = Counter()
        if self.spliter is not None:
            for dialog in tqdm.tqdm(corpus):
                for utterance in dialog:
                    c_words_ = utterance.split(self.spliter)
                    c_words = []
                    if self.char_seg:
                        for tmp in c_words_:
                            if include_chinese(tmp):
                                c_words += list(tmp)
                            else:
                                c_words.append(tmp)
                    else:
                        for tmp in c_words_:
                            c_words.append(tmp)
                    vocab_counter.update(c_words)
        else:
            for dialog in tqdm.tqdm(corpus):
                for utterance in dialog:
                    c_words_ = list(jieba.cut(utterance))
                    c_words = []
                    if self.char_seg:
                        for tmp in c_words_:
                            if include_chinese(tmp):
                                c_words += list(tmp)
                            else:
                                c_words.append(tmp)
                    else:
                        for tmp in c_words_:
                            c_words.append(tmp)
                    vocab_counter.update(c_words)

        print(f'[!] whole vocab_counter size: {len(vocab_counter)}')
        return vocab_counter

    def _build_keywords(self, corpus):
        keywords = Counter()
        for dialog in tqdm(corpus):
            for utterance in dialog:
                words = jieba.analyse.extract_tags(
                    utterance,
                    topK=self.topk,
                    allowPOS=self.allowPOS
                )
                keywords.update(words)
        print(f'[!] collect {len(keywords)} keywords')
        return keywords


class BackSeq2SeqDataset(Dataset):
    '''
    Back Seq2Seq-attn DataLoader
    '''

    def __init__(self, dataset='lccc', mode='train', lang='zh', max_length=512, max_utterance_length=150,
                 n_vocab=30000, char_seg=True):
        self.mode = mode
        # both test and train load the train_s2s.pt dataset
        assert dataset in ['lccc', 'douban']

        if dataset in ['lccc', 'douban']:
            self.spliter = " "
        self.dir_path = os.path.join(path, dataset)
        self.file_path = os.path.join(self.dir_path, f'processed', f'{mode}.txt')
        if not char_seg:
            self.pt_path = os.path.join(self.dir_path, f'{mode}.pt')
            self.vocab_path = os.path.join(self.dir_path, 'vocab.pt')
        else:
            self.pt_path = os.path.join(self.dir_path, f'{mode}_char.pt')
            self.vocab_path = os.path.join(self.dir_path, 'vocab_char.pt')
            n_vocab = 16384 - len(SPECIAL_TOKENS)

        if os.path.exists(self.pt_path):
            assert os.path.exists(self.vocab_path)
            self.vocab = torch.load(self.vocab_path)
            self.data = torch.load(self.pt_path)
            # for i in tqdm.tqdm(range(len(self.data))):
            #     self.data[i] = [np.array(self.data[i][0]), np.array(self.data[i][1]), np.array(self.data[i][2])]

            print(f'[!] load preprocessed vocab file from {self.vocab_path}, num={len(self.vocab)}')
            print(f'[!] load preprocessed {mode} file from {self.pt_path}, num={len(self.data)}')
            self.pad_id = self.vocab.vocab.stoi['[PAD]']
            self.eou_id = self.vocab.vocab.stoi['[EOU]']
            return

        print(f'[!] loading text file from {self.file_path}')
        dataset, dataset_vocab = read_text_data(self.file_path)
        print(f'[!] loading end')
        # dataset = random.sample(dataset, 500000)
        print(f'[!] generate vocab file to {self.vocab_path}')
        if self.mode == 'train':
            if lang == 'zh':
                self.vocab = ChineseTokenizer(dataset_vocab, n_vocab=n_vocab, min_freq=5, spliter=self.spliter,
                                              char_seg=char_seg)
                dataset_vocab.clear()
                self.pad_id = self.vocab.vocab.stoi['[PAD]']
                self.eou_id = self.vocab.vocab.stoi['[EOU]']
                torch.save(self.vocab, self.vocab_path)
                print(f'[!] save the vocab into {self.vocab_path}')
            else:
                raise Exception('Not adapted to the English language')
            print('[!] init the vocabulary over')
        else:
            self.vocab = torch.load(self.vocab_path)
            self.pad_id = self.vocab.vocab.stoi['[PAD]']
            self.eou_id = self.vocab.vocab.stoi['[EOU]']

        # contexts = ['[EOU]'.join(i[:-2]) for i in dataset]
        # queries = [i[-2] for i in dataset]
        # responses = [i[-1] for i in dataset]
        # nums = [len(i) for i in dataset]
        print(len(dataset))

        self.data = []
        print(f'[!] generate {mode} file to {self.pt_path}')

        for i in tqdm.tqdm(range(len(dataset))):
            dialogue = dataset[i]
            context = '[EOU]'.join(dialogue[:-2])
            query = dialogue[-2]
            response = dialogue[-1]
            num = len(dialogue)
            cid = self.vocab.encode(context.strip(), len_per_dialogue=max_length,
                                    len_per_utterance=max_utterance_length, num=num)
            qid = self.vocab.encode(query.strip(), len_per_utterance=max_utterance_length, num=2)
            rid = self.vocab.encode(response.strip(), len_per_utterance=max_utterance_length, num=1)

            bundle = [np.array(cid, dtype=np.uint16), np.array(qid, dtype=np.uint16), np.array(rid, dtype=np.uint16)]
            self.data.append(bundle)
            del dataset[i][:]
        dataset.clear()
        print(f'[!] collect {len(self.data)} samples for {mode}ing')
        torch.save(self.data, self.pt_path)
        # with open(self.pt_path, 'wb') as f:
        #     pickle.dump(self.data, f)
        print(f'[!] save the processed data into {self.pt_path}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate_lstm(self, batch):
        context = pad_sequence(
            [torch.tensor((instance[0].astype(np.int64).tolist()+instance[1].astype(np.int64).tolist()), dtype=torch.long) for instance in batch],
            batch_first=False, padding_value=self.pad_id,
        )
        context_len = torch.tensor([len(instance[0]) + len(instance[1]) for instance in batch], dtype=torch.long)

        response = pad_sequence(
            [torch.tensor(instance[1].astype(np.int64), dtype=torch.long) for instance in batch],
            batch_first=False, padding_value=self.pad_id,
        )
        response_len = torch.tensor([len(instance[2]) for instance in batch], dtype=torch.long)

        if torch.cuda.is_available():
            context, response = context.cuda(), response.cuda()
            context_len, response_len = context_len.cuda(), response_len.cuda()

        return context, context_len, response, response_len

    def collate(self, batch):
        context, context_lens, utt_lens = [], [], []
        for instance in batch:
            dialogue = []
            begin = 0
            utt_lens_per_dialogue = []
            context_str = instance[0]
            for n, tmp in enumerate(context_str):
                if tmp == self.eou_id:
                    utt = gVar(torch.tensor(context_str[begin:n + 1].astype(np.int64), dtype=torch.long))
                    dialogue.append(utt)
                    utt_lens_per_dialogue.append(n + 1 - begin)
                    begin = n + 1
            context.append(dialogue)
            context_lens.append(len(dialogue))
            utt_lens.append(gVar(torch.tensor(utt_lens_per_dialogue, dtype=torch.long)))

        context_lens = gVar(torch.tensor(context_lens, dtype=torch.long))

        query = [gVar(torch.tensor(instance[1].astype(np.int64), dtype=torch.long)) for instance in batch]
        query_len = gVar(torch.tensor([len(instance[1]) for instance in batch], dtype=torch.long))

        response = [gVar(torch.tensor(instance[2].astype(np.int64), dtype=torch.long)) for instance in batch]
        response_len = gVar(torch.tensor([len(instance[2]) for instance in batch], dtype=torch.long))

        return context, context_lens, utt_lens, query, query_len, response, response_len


remove_punc_dicts = {i: '' for i in Einglishgpunction + Chinesepunction}
remove_punc_table = str.maketrans(remove_punc_dicts)

def remove_pounc(string):
    new_s = string.translate(remove_punc_table)
    return new_s


class NonParallelDataset(Dataset):
    '''
    Non-Parallel Dataset DataLoader
    '''

    def __init__(self, non_parallel_dataset='div_non_conv', parallel_dataset='lccc',
                 max_utterance_length=150, char_seg=True):

        assert non_parallel_dataset in ['div', 'inf', ]
        assert parallel_dataset in ['lccc', 'douban']
        self.dir_path = os.path.join(path, non_parallel_dataset)
        self.file_path = os.path.join(self.dir_path, f'processed', 'train.txt')

        if char_seg:
            self.pt_path = os.path.join(self.dir_path, f'train_char.pt')
            self.vocab_path = os.path.join(path, parallel_dataset, 'vocab_char.pt')
        else:
            self.pt_path = os.path.join(self.dir_path, f'train.pt')
            self.vocab_path = os.path.join(path, parallel_dataset, 'vocab.pt')

        if parallel_dataset in ['douban']:
            self.non_pounc = True
            self.pt_path = os.path.join(self.dir_path, f'train_char_wo_pounc.pt')
        else:
            self.non_pounc = False

        if os.path.exists(self.pt_path):
            assert os.path.exists(self.vocab_path)
            self.vocab = torch.load(self.vocab_path)
            print(f'[!] load preprocessed vocab file from {self.vocab_path}, num={len(self.vocab)}')
            self.data = torch.load(self.pt_path)
            # with open(self.pt_path, 'rb') as f:
            #     self.data = pickle.load(f)
            print(f'[!] load preprocessed train file from {self.pt_path}, num={len(self.data)}')
            self.pad_id = self.vocab.vocab.stoi['[PAD]']

            return

        dataset, _ = read_text_data(self.file_path)

        print(f'[!] read_text_data {self.file_path}')
        self.vocab = torch.load(self.vocab_path)
        self.pad_id = self.vocab.vocab.stoi['[PAD]']

        utterances = [i[0] for i in dataset]

        print(f'[!] collect {len(utterances)} samples in {self.file_path}')
        self.data = []
        for utterance in \
                tqdm.tqdm(utterances):
            # print(utterance)
            if self.non_pounc:
                utterance = remove_pounc(utterance).strip()
            else:
                utterance = utterance.strip()

            cid = self.vocab.encode(utterance, len_per_utterance=max_utterance_length, num=1)
            self.data.append(np.array(cid, dtype=np.uint16))
        print(f'[!] collect {len(self.data)} samples for training')

        torch.save(self.data, self.pt_path)
        # with open(self.pt_path, 'wb') as f:
        #     pickle.dump(self.data, f)
        print(f'[!] save the processed data into {self.pt_path}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate(self, batch):
        utterance = [gVar(torch.tensor(instance.astype(np.int64), dtype=torch.long)) for instance in batch]
        utterance_len = gVar(torch.tensor([len(instance) for instance in batch], dtype=torch.long))

        return utterance, utterance_len


if __name__ == '__main__':
    import random

    torch.cuda.manual_seed_all(2021)
    random.seed(2021)
    torch.manual_seed(2021)
    np.random.seed(2021)
    PDataset = BackSeq2SeqDataset(max_utterance_length=50, char_seg=False)
    NPDataset = NonParallelDataset(max_utterance_length=50, char_seg=False)

    PDataset_char = BackSeq2SeqDataset(max_utterance_length=150, char_seg=True)
    NPDataset_char = NonParallelDataset(max_utterance_length=150, char_seg=True)

    PDataloader = DataLoader(PDataset, shuffle=True, batch_size=8, collate_fn=PDataset.collate)
    NPDataloader = DataLoader(NPDataset, shuffle=True, batch_size=8, collate_fn=NPDataset.collate)

    for n, data in enumerate(PDataloader):
        # print(n)
        # context, context_len, query, query_len, response, response_len = data
        context, context_lens, utt_lens, query, query_len, response, response_len = data
        num = len(context)
        print(f'batch size {len(context)}')
        for i in range(num):
            for j in range(len(context[i])):
                print('size:', context[i][j].size())
                if len(context[i][j]) > 0:
                    print('context:', PDataset.vocab.decode(context[i][j]))
            print('query:', PDataset.vocab.decode(query[i]))
            print('response:', PDataset.vocab.decode(response[i]))
        if n == 1:
            break

    for n, data in enumerate(NPDataloader):
        # print(n)
        utterance_batch, utterance_len = data
        # print('batch_size:', utterance_len.size(), 'length:', utterance_len)
        for i in range(utterance_len.size(0)):
            sample = utterance_batch[:, i]
            NPDataset.vocab.decode(sample)
            print(NPDataset.vocab.decode(sample))
        # print(utterance_batch.size())

        # print(NPDataloader.vocab.decode(utterance_batch))
        # print(utterance_len)
        if n == 1:
            break
