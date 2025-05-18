import re
import argparse

def load_bpe_codes(bpe_codes_file):
    with open(bpe_codes_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    bpe_codes = [tuple(line.strip().split())[:2] for line in lines]
    return bpe_codes

def apply_bpe(word, bpe_codes):
    word = list(word) + ['</w>']
    i = 0
    while i < len(word)-1:
        pair = (word[i], word[i+1])
        if pair in bpe_codes:
            word = word[:i] + [''.join(pair)] + word[i+2:]
            i = max(i-1, 0)
        else:
            i += 1
    return ' '.join(word)

def encode_file(input_file, output_file, bpe_codes):
    bpe_codes_set = set(bpe_codes)
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                fout.write('\n')
                continue
            words = line.strip().split()
            bpe_words = [apply_bpe(word, bpe_codes_set) for word in words]
            fout.write(' '.join(bpe_words) + '\n')
def apply_bpe_file(bpe_codes_file,input_file,output_file):
    bpe_codes = load_bpe_codes(bpe_codes_file)
    encode_file(input_file,output_file,bpe_codes)

if __name__ == "__main__":
    #python apply_bpe.py --input data/train.txt --output data/train_bpe.txt --bpe_codes data/bpe_codes_20000.txt
    #python apply_bpe.py --input data/dev.txt --output data/dev_bpe.txt --bpe_codes data/bpe_codes_20000.txt
    parser = argparse.ArgumentParser(description='应用BPE编码到数据集。')
    parser.add_argument('--input', type=str, required=True, help='输入数据文件路径（例如，train.txt）')
    parser.add_argument('--output', type=str, required=True, help='保存BPE编码后的数据文件路径')
    parser.add_argument('--bpe_codes', type=str, required=True, help='BPE代码文件路径')
    args = parser.parse_args()

    bpe_codes = load_bpe_codes(args.bpe_codes)
    encode_file(args.input, args.output, bpe_codes)