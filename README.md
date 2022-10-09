# Multi-turn Dialog  Generation


## Dataset 
The preprocess script for these datasets can be found under `data/` folder.
1. DailyDialog dataset
2. Ubuntu corpus
5. PersonaChat

## Metric
1. PPL: test perplexity
2. BLEU(1-4):  
3. Embedding-based metrics: Average, Extrema, Greedy  
4. Distinct-1/2


## Requirements
1. Pytorch 1.2+  
2. Python 3.6.1+
3. tqdm
4. numpy
5. nltk 3.4+
6. scipy
7. sklearn (optional)
9. **GoogleNews word2vec** or **glove 300 word2vec** (optional)
10. tensorboard (for PyTorch 1.2+)

## Dataset format
Three multi-turn open-domain dialogue dataset (Dailydialog, PersonaChat, UbuntuV2) 
Dailydialog and PersonaChat can be obtained by this [link](https://github.com/ZihaoW123/PHAED/raw/main/data/data.zip)
UbuntuV2 can be obtained by this [link](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)

Each dataset contains 6 files
* src-train.txt
* tgt-train.txt
* src-dev.txt
* tgt-dev.txt
* src-test.txt
* tgt-test.txt

In all the files, one line contain only one dialogue context (src) or the dialogue response (tgt).
More details can be found in the example files.
In order to create the graph, each sentence must begin with the 
special tokens `<user0>` and `<user1>` which denote the speaker.
The `__eou__` is used to separate the multiple utterances in the conversation context.
More details can be found in the small data case.

## How to use

* Model names: `PHAED`
* Dataset names: `DaildyDialog, PersonaChat, Ubuntu`

### 0. Ready
Before running the following commands, make sure the essential folders are created:
```bash
mkdir -p processed/$DATASET
mkdir -p data/$DATASET
mkdir -p tblogs/$DATASET
mkdir -p ckpt/$DATASET
```

Variable `DATASET` contains the name of the dataset that you want to process


### 1. Generate the vocab of the dataset

```bash
./run.sh vocab <dataset>
```

```bash
# example: get the vocab of DailyDialog dataset
./run.sh vocab DailyDialog
```

### 2. Train the model on corresponding dataset

```bash
./run.sh train <dataset> <model> <cuda>
```

```bash
# example: train the PHAED model with DailyDialog dataset on 0th GPU
./run.sh train DailyDialog PHAED 0
```

### 3. Translate the test dataset:

```bash
./run.sh translate <dataset> <model> <cuda>
```

```bash
# example: generation the response. translate mode, dataset dialydialog, model PHAED on 0th GPU
./run.sh translate DailyDialog PHAED 0
```
 

### 4. Evaluate the result of the translated utterances

```bash
./run.sh eval <dataset> <model> <cuda>
```
 
```bash
# example: get the BLEU, Distinct, embedding-based metrics result of the generated sentences on 0th GPU
./run.sh eval DailyDialog PHAED 0
```
