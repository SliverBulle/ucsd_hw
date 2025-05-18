import re
import collections
import argparse
from sentiment_data import read_sentiment_examples
from tqdm import tqdm
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def train_bpe(input_file, num_merges, output_codes):
    # 读取训练数据并构建初始词汇表
    examples = read_sentiment_examples(input_file)
    vocab = collections.defaultdict(int)
    for ex in examples:
        for word in ex.words:
            # 在每个单词末尾添加 '</w>' 标记
            word = ' '.join(list(word)) + ' </w>'
            vocab[word] += 1
    bpe_codes = []
    # 迭代进行BPE合并
    for i in tqdm(range(num_merges), desc='BPE merging'):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        bpe_codes.append(best)

    # 保存BPE合并规则
    with open(output_codes, 'w', encoding='utf-8') as f:
        for pair in bpe_codes:
            f.write(f'{pair[0]} {pair[1]}\n')

if __name__ == "__main__":
    #python bpe_encoder.py --input data/train.txt --num_merges 20000 --output data/bpe_codes_20000.txt
    parser = argparse.ArgumentParser(description='训练BPE模型。')
    parser.add_argument('--input', type=str, required=True, help='训练数据文件路径（train.txt）')
    parser.add_argument('--num_merges', type=int, required=True, help='BPE合并次数（词汇表大小）')
    parser.add_argument('--output', type=str, required=True, help='保存BPE代码的文件路径')
    args = parser.parse_args()

    train_bpe(args.input, args.num_merges, args.output)