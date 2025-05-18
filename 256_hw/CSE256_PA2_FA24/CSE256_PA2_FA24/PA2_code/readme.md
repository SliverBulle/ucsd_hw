# CSE256 PA2 - Transformer with AliBi

## structure

```
PA2_code/
├── alibi.py
├── alibi_decoder.py
├── adapted_transformer.py
├── decoder.py
├── encoder.py
├── dataset.py 
├── transformer.py
├── tokenizer.py
├── utilities.py 
├── readme.md
└── plot/
└── model/

```


## environment

- Python 3.8+
- PyTorch 1.8+
- NLTK
- Matplotlib

## install dependencies

```
pip install torch, nltk, matplotlib
```


## run

1. create necessary directories:
```
mkdir plot model
```

2. train standard transformer encoder:
```
python encoder.py
```
3. train standard transformer decoder:
```
python decoder.py
```
4. train alibi transformer decoder:
```
python alibi_decoder.py
```
5. train adapted transformer decoder:
```
python adapted_transformer.py
```

## training results

Training process will generate the following files:

1. standard transformer:
- `plot/training_perplexity_{current_time}.png`: 
- `plot/transformer_test_perplexity_{current_time}.png`: 
- `model/decoder.pth`: 保存的模型

2. AliBi Transformer:
- `plot/alibi_decoder_train_perplexity_{current_time}.png`: 
- `plot/alibi_decoder_test_perplexity_{current_time}.png`: 
- `model/alibi_decoder.pth`: 保存的模型

## code reference

- AliBi implementation: /PA2_code/adapted_transformer.py
- transformer implementation: /PA2_code/transformer.py


## notes

1. Ensure the `speechesdataset` folder is in the correct location
2. Create `plot` and `model` directories before running
3. GPU training requires CUDA support
4. During training, perplexity will be displayed, which can be visually inspected through the generated charts
