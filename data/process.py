import tqdm
import ipdb
import csv

'''
Download PersonaChat / Dailydialog / DSTC7_AVSD dataset from: https://github.com/PaddlePaddle/models/tree/75e463a22ef6cbd43f47917a62ee43d13a05831e/PaddleNLP/Research/Dialogue-PLATO

# DATASET: DailyDialog, PersonaChat, DSTC7_AVSD
# MODE: train, test, valid
# after processing, remember to rename the valid files to dev files (src-dev.txt, tgt-dev.txt)
python plato_process.py $DATASET $MODE
'''


def load_file(path):
    corpus = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            corpus.append(line.strip())
    return corpus


def write_file_dailydialog(corpus, dataset, mode):
    src, tgt = [], []
    for dialogue in corpus:
        se = dialogue.split('\t')
        assert len(se) == 2
        context, response = se
        utterances = context.split('__eou__')
        utterances = ['<user0> ' + i.strip() if idx % 2 == 0 else '<user1> ' + i.strip() for idx, i in enumerate(utterances)]
        last_speaker = utterances[-1][:7]
        response = '<user1> ' + response if last_speaker == '<user0>' else '<user0> ' + response
        utterances = ' __eou__ '.join(utterances)
        src.append(utterances)
        tgt.append(response)
    num = len(src)
    src_new = []
    tgt_new = []
    for i in range(num):
        tmp_src = src[i]
        tmp_tgt = tgt[i]
        if i+2 < num and src[i+2].find(tmp_src) != -1:
            #print('i:',i, num)
            continue
        #elif i+1 < num and src[i+1].find(tmp_src) != -1:
        #    continue
        src_new.append(tmp_src)
        tgt_new.append(tmp_tgt)
    
    if mode == 'valid':
        mode='dev'

    if mode == 'train':
        src = src_new
        tgt = tgt_new

    src_path = f'{dataset}/src-{mode}.txt'
    tgt_path = f'{dataset}/tgt-{mode}.txt'
    with open(src_path, 'w', encoding='UTF-8') as f:
        for i in src:
            f.write(f'{i}\n')

    with open(tgt_path, 'w', encoding='UTF-8') as f:
        for i in tgt:
            f.write(f'{i}\n')
            
            
def write_file_personachat(corpus, dataset, mode):
    src, tgt = [], []
    for dialogue in corpus:
        se = dialogue.split('\t')
        assert len(se) == 3
        knowledge, context, response = se
        ku = knowledge.split('__eou__') 
        
        ku = ['<user0> ' + i.strip() for i in ku]
        ku = ' __eou__ '.join(ku)

        cu = reversed(context.split('__eou__'))
        response = '<user0> ' + response

        speaker = '<user1> '
        fcu = []
        for i in cu:
            fcu.append(speaker + i.strip())
            speaker = '<user0> ' if speaker == '<user1> ' else '<user1> '
        fcu = ' __eou__ '.join(list(reversed(fcu)))
        src.append(ku + ' __eou__ ' + fcu)
        tgt.append(response)
    num = len(src)
    src_new = []
    tgt_new = []
    for i in range(num):
        tmp_src = src[i]
        tmp_tgt = tgt[i]
        if i + 2 < num and src[i + 2].find(tmp_src) != -1:
            continue
        #elif i + 1 < num and src[i + 1].find(tmp_src) != -1:
        #    continue
        src_new.append(tmp_src)
        tgt_new.append(tmp_tgt)

    if mode == 'train':
        src = src_new
        tgt = tgt_new

    if mode == 'valid':
        mode='dev'
    src_path = f'{dataset}/src-{mode}.txt'
    tgt_path = f'{dataset}/tgt-{mode}.txt'
    with open(src_path, 'w', encoding='UTF-8') as f:
        for i in src:
            f.write(f'{i}\n')

    with open(tgt_path, 'w', encoding='UTF-8') as f:
        for i in tgt:
            f.write(f'{i}\n')
            
def write_file_ubuntu(corpus, dataset, mode):
    src, tgt = [], [] 
    if mode == 'valid':
        mode='dev'
    for example in corpus[1:]:
        dialogue = example[0].strip() + " " +  example[1].strip() 
        #print('1  ', dialogue)
        dialogue = dialogue.strip().split(' __eot__')
        #print('2  ', dialogue)
        #print(' ',  ) 
        all = []
        for cur_u in dialogue:
            if len(cur_u.strip())>0: 
                tmp = cur_u.replace(' __eou__', '').strip()
                all.append(tmp[:])
                if len(all)>=2:
                    response = '<user0> ' + all[-1][:]
                    utterances = []
                    user = '<user1> '
                    for i in range(len(all)-2, -1, -1): 
                        utterances = [user + all[i][:]] +  utterances
                        user = '<user1> ' if user == '<user0> ' else '<user0> ' 
                    utterances = ' __eou__ '.join(utterances)
                    src.append(utterances)
                    tgt.append(response) 
    

    if mode == 'train':
        num = len(src)
        src_new = []
        tgt_new = []
        for i in range(num):
            tmp_src = src[i]
            tmp_tgt = tgt[i]
            if i + 2 < num and src[i + 2].find(tmp_src) != -1:
                continue
            #elif i + 1 < num and src[i + 1].find(tmp_src) != -1:
            #    continue
            src_new.append(tmp_src)
            tgt_new.append(tmp_tgt) 
        src = src_new
        tgt = tgt_new

    src_path = f'{dataset}/src-{mode}.txt'
    tgt_path = f'{dataset}/tgt-{mode}.txt'
    with open(src_path, 'w', encoding='UTF-8') as f:
        for i in src:
            f.write(f'{i}\n')

    with open(tgt_path, 'w', encoding='UTF-8') as f:
        for i in tgt:
            f.write(f'{i}\n')
            

def load_csv(path): 
    corpus = []
    i = 0
    with open(path, 'r', encoding='UTF-8') as f:
        for line in csv.reader(f, skipinitialspace=True):
            corpus.append(line)  
            i+=1  
    return corpus


if __name__ == "__main__":
    import sys
    dataset = sys.argv[1]

    if dataset == 'DailyDialog':
        write_file_fuchtion = write_file_dailydialog
    elif dataset == 'PersonaChat':  
        write_file_fuchtion = write_file_personachat
    elif dataset == 'Ubuntu':
        write_file_fuchtion = write_file_ubuntu  
    else:
        raise Exception('[!] obtain the wrong dataset {dataset}')
        
    if dataset in ['DailyDialog', 'PersonaChat']: 
        mode = 'train'
        corpus = load_file(f'{dataset}/dial.{mode}')
        write_file_fuchtion(corpus, dataset, mode)
        mode = 'valid'
        corpus = load_file(f'{dataset}/dial.{mode}')
        write_file_fuchtion(corpus, dataset, mode)
        mode = 'test'
        corpus = load_file(f'{dataset}/dial.{mode}')
        write_file_fuchtion(corpus, dataset, mode)
    elif dataset in ['Ubuntu']:  
        mode = 'train'
        corpus = load_csv(f'Ubuntu/ubuntu-ranking-dataset-creator/src/{mode}.csv')
        write_file_fuchtion(corpus, dataset, mode)
        
        mode = 'valid'
        corpus = load_csv(f'Ubuntu/ubuntu-ranking-dataset-creator/src/{mode}.csv')
        write_file_fuchtion(corpus, dataset, mode)
        
        mode = 'test'
        corpus = load_csv(f'Ubuntu/ubuntu-ranking-dataset-creator/src/{mode}.csv')
        write_file_fuchtion(corpus, dataset, mode)
