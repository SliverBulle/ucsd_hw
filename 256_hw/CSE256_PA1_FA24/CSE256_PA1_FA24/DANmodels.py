# DANmodels.py

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from sentiment_data import read_sentiment_examples
from collections import defaultdict
from torchviz import make_dot

class SentimentDatasetDAN(Dataset):
    """
    Dataset class for handling sentiment analysis data using Deep Averaging Network (DAN).
    
    This class converts sentences into sequences of word indices based on a provided vocabulary.
    It handles padding and unknown words.
    
    Attributes:
        examples (List[SentimentExample]): List of sentiment examples.
        vocab (dict): Mapping from words to their corresponding indices.
        max_length (int): Maximum number of words per sentence.
        sentences (List[torch.Tensor]): List of padded word index tensors.
        labels (torch.Tensor): Tensor of labels.
    """
    def __init__(self, infile: str, vocab: dict, max_length: int = 50):
        """
        Initializes the dataset by reading the data, processing sentences, and preparing tensors.
        
        :param infile: Path to the input file containing sentiment data.
        :param vocab: Dictionary mapping words to indices.
        :param max_length: Maximum number of words per sentence.
        """
        self.examples = read_sentiment_examples(infile)
        self.vocab = vocab
        self.max_length = max_length
        self.sentences = [self.preprocess(ex.words) for ex in self.examples]
        self.labels = torch.tensor([ex.label for ex in self.examples], dtype=torch.long)

    def preprocess(self, words: list) -> torch.Tensor:
        """
        Converts a list of words into a padded tensor of word indices.
        
        :param words: List of words in the sentence.
        :return: Padded tensor of word indices.
        """
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in words[:self.max_length]]
        # Pad the sequence if it's shorter than max_length
        if len(indices) < self.max_length:
            indices += [self.vocab['<PAD>']] * (self.max_length - len(indices))
        return torch.tensor(indices, dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]


class DAN(nn.Module):
    """
    Deep Averaging Network (DAN) for sentiment analysis.
    
    This model averages word embeddings and passes the result through a fully connected network.
    
    Attributes:
        embedding (nn.Embedding): Embedding layer initialized with pre-trained GloVe embeddings.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer (output layer).
        dropout (nn.Dropout): Dropout layer to prevent overfitting.
        log_softmax (nn.LogSoftmax): Log-Softmax activation for output.
    """
    def __init__(self, embedding_matrix: torch.Tensor, hidden_dim: int = 100, output_dim: int = 2, dropout_rate: float = 0.5):
        """
        Initializes the DAN model with the given parameters.
        
        :param embedding_matrix: Pre-trained GloVe embedding matrix.
        :param hidden_dim: Dimension of the hidden layer.
        :param output_dim: Number of output classes.
        :param dropout_rate: Dropout rate.
        """
        super(DAN, self).__init__()
        num_embeddings, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight = nn.Parameter(embedding_matrix, requires_grad=True)  # Allow fine-tuning
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.log_softmax = nn.LogSoftmax(dim=1)
        #self.l2_regularization = 1e-5

    def forward(self, x):
        """
        Forward pass for the DAN model.
        
        :param x: Input tensor of word indices with shape (batch_size, max_length).
        :return: Log-Softmax probabilities with shape (batch_size, output_dim).
        """
        embedded = self.embedding(x)  # Shape: (batch_size, max_length, embedding_dim)
        averaged = embedded.mean(dim=1)  # Shape: (batch_size, embedding_dim)
        hidden = F.relu(self.fc1(averaged))  # Shape: (batch_size, hidden_dim)
        dropped = self.dropout(hidden)  # Shape: (batch_size, hidden_dim)
        output = self.fc2(dropped)  # Shape: (batch_size, output_dim)
        #l2_loss = self.l2_regularization * (self.fc1.weight.norm(2) + self.fc2.weight.norm(2))
        return self.log_softmax(output)


def build_vocab(examples: list, min_freq: int = 1) -> dict:
    """
    Builds a vocabulary dictionary mapping words to unique indices.
    
    :param examples: List of SentimentExample instances.
    :param min_freq: Minimum frequency for a word to be included in the vocabulary.
    :return: Dictionary mapping words to indices.
    """
    counter = defaultdict(int)
    for ex in examples:
        for word in ex.words:
            counter[word] += 1
    #objs_to_ints
    # Start indexing from 2 to reserve 0 for <PAD> and 1 for <UNK>
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab


def load_glove_embeddings(file_path: str, vocab: dict, embedding_dim: int = 300) -> torch.Tensor:
    """
    Loads pre-trained GloVe embeddings and creates an embedding matrix aligned with the vocabulary.
    
    :param file_path: Path to the GloVe embeddings file.
    :param vocab: Vocabulary dictionary mapping words to indices.
    :param embedding_dim: Dimension of the GloVe embeddings.
    :return: Embedding matrix as a PyTorch tensor.
    """
    embeddings = np.random.normal(scale=0.6, size=(len(vocab), embedding_dim))  # Initialize with random vectors
    embeddings[vocab['<PAD>']] = np.zeros(embedding_dim)  # <PAD> is a zero vector

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            split = line.strip().split()
            word = split[0]
            if word in vocab:
                vector = np.array(split[1:], dtype=np.float32)
                embeddings[vocab[word]] = vector

    return torch.tensor(embeddings, dtype=torch.float32)

