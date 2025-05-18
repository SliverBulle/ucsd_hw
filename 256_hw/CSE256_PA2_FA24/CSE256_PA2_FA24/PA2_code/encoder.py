import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import torch.nn as nn
import torch.optim as optim
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
import matplotlib.pyplot as plt
from transformer import TransformerEncoder, TransformerDecoder, FeedforwardClassifier

import nltk
seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers

eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set

## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15  # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts


def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels


def compute_classifier_accuracy(encoder, classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader using the encoder."""
    encoder.eval()
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            embeddings, _ = encoder(X)  # input to encoder
            outputs = classifier(embeddings)  # output of classifier
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
    accuracy = (100 * total_correct / total_samples)
    encoder.train()
    classifier.train()
    return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses = []
    total_loss = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            logits, _ = decoderLMmodel(X)  # get decoder output
            # rearrange logits and Y to match CrossEntropyLoss input format
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            Y = Y.view(-1)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, Y)
            losses.append(loss.item())
            total_loss += loss.item()
            if len(losses) >= eval_iters:
                break

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity


def main():
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts))  # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)

    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    # initialize model
    encoder = TransformerEncoder(
        vocab_size=tokenizer.vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=0.1
    ).to(device)

    classifier = FeedforwardClassifier(n_input, n_hidden, n_output).to(device)

    # define loss function and optimizer
    criterion_cls = nn.CrossEntropyLoss()
    optimizer_cls = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate)

    # train classifier and encoder
    train_accuracies = []
    for epoch in range(epochs_CLS):
        encoder.train()
        classifier.train()
        total_loss = 0
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer_cls.zero_grad()

            # forward propagation
            embeddings, _ = encoder(xb)  # get encoder output
            outputs = classifier(embeddings)  # classifier output

            loss = criterion_cls(outputs, yb)
            loss.backward()
            optimizer_cls.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_CLS_loader)
        accuracy = compute_classifier_accuracy(encoder, classifier, train_CLS_loader)
        print(f"Epoch [{epoch+1}/{epochs_CLS}], Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.2f}%")
        
        train_accuracies.append(accuracy)

    # compute test accuracy
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=False)
    test_accuracy = compute_classifier_accuracy(encoder, classifier, test_CLS_loader)
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    plt.figure(figsize=(8, 6))
    plt.plot(train_accuracies, label='Transformer')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy for Transformer')
    plt.legend()
    plt.grid()

    # save training accuracy plot
    import time
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")   
    training_accuracy_file = f'plot/transformer_train_accuracy_{current_time}.png'
    plt.savefig(training_accuracy_file)
    print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

    # compute number of parameters in encoder and classifier
    total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in classifier.parameters())
    print(f"Total number of parameters in encoder and classifier: {total_params}")



    # sanity check
    from utilities import Utilities
    utility = Utilities(tokenizer, encoder)
    test_sentence = "I don't know about you, but I'm not ready to take a ten percent chance on change."
    utility.sanity_check(test_sentence, block_size)

if __name__ == "__main__":
    main()
