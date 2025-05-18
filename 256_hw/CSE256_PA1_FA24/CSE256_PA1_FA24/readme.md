
# CSE256 PA1 FA24 - Sentiment Analysis Project
## Project Overview

This project implements a sentiment analysis system using Deep Averaging Networks (DAN) and Bag-of-Words (BOW) models. It utilizes different word embedding methods, including pre-trained GloVe embeddings, randomly initialized embeddings, and Byte Pair Encoding (BPE) for text processing.

## Installation

```bash
pip install torch numpy matplotlib tqdm scikit-learn torchviz
```

or

```bash
pip install -r requirements.txt
```

## Usage
1. Train DAN model (using GloVe embeddings):
```bash
python main.py --model DAN
```

2. Train DAN model (randomly initialized embeddings):
```bash
python main.py --model DAN_random
```

3. Train DAN model (BPE embeddings):
merge_num is the number of BPE merges,if you don't specify it, default is 20000.
```bash
python main.py --model DAN_BPE --merge_num 20000
```
Be sure you have `data`, `plots` and `models` folder in the root directory.
## Model Architectures

### DAN Model

The DAN model implementation can be found in the `DANmodels.py` file


### BOW Model

The BOW model implementations can be found in the `BOWmodels.py` file


## Experimental Results


The training process generates accuracy curves, saved as PNG files in the plots folder.
Weights are saved in the models folder.


## Main Program Structure

- `main.py`: Main script for training and evaluating the models.
- `sentiment_data.py`: Utility for reading sentiment data.
- `DANmodels.py`: Contains the implementation of DAN and BOW models.
- `bpe_encoder.py`: Implements BPE encoding for text processing.
- `apply_bpe.py`: Applies BPE encoding to the input text.

## Contributors
- Lei Hu