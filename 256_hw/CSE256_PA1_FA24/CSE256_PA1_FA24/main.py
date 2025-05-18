import argparse
import time
import torch
from torch.utils.data import DataLoader
from DANmodels import SentimentDatasetDAN, DAN, build_vocab, load_glove_embeddings
from sentiment_data import read_sentiment_examples
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchviz import make_dot
import re
import collections
from apply_bpe import apply_bpe_file
from bpe_encoder import train_bpe
import datetime

def train_epoch(data_loader, model, loss_fn, optimizer):
    model.train()
    train_loss, correct = 0, 0
    for X, y in data_loader:
        X = X.long()  # Ensure the input is converted to LongTensor
        y = y.long()
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    accuracy = correct / len(data_loader.dataset)
    return accuracy, train_loss / len(data_loader)

def eval_epoch(data_loader, model, loss_fn, optimizer):
    model.eval()
    eval_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X = X.long()  # Ensure the input is converted to LongTensor
            y = y.long()
            pred = model(X)
            loss = loss_fn(pred, y)
            eval_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    accuracy = correct / len(data_loader.dataset)
    return accuracy, eval_loss / len(data_loader)

def experiment(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Add learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        scheduler.step(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy
    # Move data to device
def move_to_device(batch, device):
    X, y = zip(*batch)  # Split batch data into X and y
    X = torch.stack(X).to(device)  # Combine X into a tensor and move to device
    y = torch.tensor(y).to(device)  # Convert y to tensor and move to device
    return X, y

def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., DAN, DAN_random, SUBWORDDAN)')
    parser.add_argument('--bpe_codes', type=str, required=False, help='BPE codes file path')
    parser.add_argument('--merge_num', type=int, required=False, help='Number of BPE merges (only for SUBWORDDAN)')
    args = parser.parse_args()

    if args.model == "DAN":

        # Load training and development data
        train_examples = read_sentiment_examples("data/train.txt")
        dev_examples = read_sentiment_examples("data/dev.txt")

        # Build vocabulary
        vocab = build_vocab(train_examples)
        print(f"Vocabulary size: {len(vocab)}")

        # Load pre-trained GloVe embeddings
        embedding_matrix = load_glove_embeddings('data/glove.6B.300d-relativized.txt', vocab, embedding_dim=300)
        print(f"Embedding matrix shape: {embedding_matrix.shape}")

        # Create datasets and data loaders
        train_data = SentimentDatasetDAN("data/train.txt", vocab, max_length=50)
        dev_data = SentimentDatasetDAN("data/dev.txt", vocab, max_length=50)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

        # Initialize DAN model
        model = DAN(embedding_matrix, hidden_dim=100, output_dim=2, dropout_rate=0.2)

        # Move model to available device (GPU if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)


        train_loader = DataLoader(
            train_data,
            batch_size=16,
            shuffle=True,
            collate_fn=lambda x: move_to_device(x, device)  # Pass entire batch and device
        )
        test_loader = DataLoader(
            dev_data,
            batch_size=16,
            shuffle=False,
            collate_fn=lambda x: move_to_device(x, device)  # Pass entire batch and device
        )

        # Train and evaluate the model
        start_time = time.time()
        dan_train_accuracy, dan_test_accuracy = experiment(model, train_loader, test_loader)
        end_time = time.time()
        
            # Plot training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_train_accuracy, label='DAN')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for DAN')
        plt.legend()
        plt.grid()

        # Save training accuracy plot
        training_accuracy_file = 'dan_train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot development accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_test_accuracy, label='DAN')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for DAN')
        plt.legend()
        plt.grid()

        # Save development accuracy plot
        testing_accuracy_file = 'dan_dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")
        print(f"Training completed in {end_time - start_time:.2f} seconds.")
    elif args.model == "DAN_random":

            # Load training and development data
        train_examples = read_sentiment_examples("data/train.txt")

        # Build vocabulary
        vocab = build_vocab(train_examples)
        print(f"Vocabulary size: {len(vocab)}")

        # Initialize embedding matrix randomly
        embedding_dim = 300
        embedding_matrix = torch.randn(len(vocab), embedding_dim)
        print(f"Embedding matrix shape: {embedding_matrix.shape}")


        # Create datasets and data loaders
        train_data = SentimentDatasetDAN("data/train.txt", vocab, max_length=50)
        dev_data = SentimentDatasetDAN("data/dev.txt", vocab, max_length=50)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

        # Initialize DAN model
        model = DAN(embedding_matrix, hidden_dim=100, output_dim=2, dropout_rate=0.2)

        # Move model to available device (GPU if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)


        train_loader = DataLoader(
            train_data,
            batch_size=16,
            shuffle=True,
            collate_fn=lambda x: move_to_device(x, device)  # Pass entire batch and device
        )
        test_loader = DataLoader(
            dev_data,
            batch_size=16,
            shuffle=False,
            collate_fn=lambda x: move_to_device(x, device)  # Pass entire batch and device
        )

        # Train and evaluate the model
        start_time = time.time()
        dan_train_accuracy, dan_test_accuracy = experiment(model, train_loader, test_loader)
        end_time = time.time()
        
            # Plot training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_train_accuracy, label='DAN')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for DAN')
        plt.legend()
        plt.grid()

        # Save training accuracy plot
        training_accuracy_file = 'dan_random_init_train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot development accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_test_accuracy, label='DAN')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for DAN')
        plt.legend()
        plt.grid()

        # Save development accuracy plot
        testing_accuracy_file = 'dan_random_init_dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")
        print(f"Training completed in {end_time - start_time:.2f} seconds.")
    elif args.model == "SUBWORDDAN":
        start_time = time.time()
        if args.merge_num is None:
            merge_num = 20000
        else:
            merge_num = args.merge_num
        # Generate file names with merge_num
        bpe_codes_file = f"data/bpe_codes_{merge_num}.txt"
        train_bpe_file = f"data/train_bpe_{merge_num}.txt"
        dev_bpe_file = f"data/dev_bpe_{merge_num}.txt"

        # Train BPE and generate BPE codes
        print("Starting BPE encoding training...")
        train_bpe("data/train.txt", merge_num, bpe_codes_file)
        print(f"BPE encoding training completed, BPE codes saved in {bpe_codes_file}")

        # Apply BPE to training and development data
        print("Applying BPE to dataset...")
        apply_bpe_file(bpe_codes_file=bpe_codes_file,input_file="data/train.txt",output_file=train_bpe_file)
        apply_bpe_file(bpe_codes_file=bpe_codes_file,input_file="data/dev.txt",output_file=dev_bpe_file)
        print(f"BPE applied, generated {train_bpe_file} and {dev_bpe_file}")

        # Load training and development data (BPE encoded)
        train_examples = read_sentiment_examples(train_bpe_file)
        dev_examples = read_sentiment_examples(dev_bpe_file)

        # Build subword vocabulary
        vocab = build_vocab(train_examples)
        print(f"Subword Vocabulary size: {len(vocab)}")

        # Initialize embedding matrix randomly
        embedding_dim = 300
        embedding_matrix = torch.randn(len(vocab), embedding_dim)
        print(f"Embedding matrix shape: {embedding_matrix.shape}")

        # Create datasets and data loaders
        train_data = SentimentDatasetDAN(train_bpe_file, vocab, max_length=50)
        dev_data = SentimentDatasetDAN(dev_bpe_file, vocab, max_length=50)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

        # Initialize DAN model
        model = DAN(embedding_matrix, hidden_dim=100, output_dim=2, dropout_rate=0.2)

        # Move model to available device (GPU if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        train_loader = DataLoader(
            train_data,
            batch_size=16,
            shuffle=True,
            collate_fn=lambda x: move_to_device(x, device)  # Pass entire batch and device
        )
        test_loader = DataLoader(
            dev_data,
            batch_size=16,
            shuffle=False,
            collate_fn=lambda x: move_to_device(x, device)  # Pass entire batch and device
        )

        # Train and evaluate the model
        dan_train_accuracy, dan_test_accuracy = experiment(model, train_loader, test_loader)
        end_time = time.time()
        
        # Plot training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_train_accuracy, label='DAN BPE')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for DAN with BPE')
        plt.legend()
        plt.grid()

        # Save training accuracy plot
        training_accuracy_file = f'plots/dan_bpe_train_accuracy_{merge_num}.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot development accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_test_accuracy, label='DAN BPE')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for DAN with BPE')
        plt.legend()
        plt.grid()

        # Save development accuracy plot
        testing_accuracy_file = f'plots/dan_bpe_dev_accuracy_{merge_num}.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")
        print(f"Training completed in {end_time - start_time:.2f} seconds.")
    elif args.model == "DAN_BPE_5000":
        # Similar to DAN_BPE but with 5000 merge operations
        # Modify input file and BPE codes path
        pass
    # More BPE vocabulary sizes can be added as needed
    else:
        print("Unsupported model type. Please choose from DAN, DAN_random, SUBWORDDAN.")
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), f'models/model_weights_{args.model}_{date_str}.pth')
    print(f"Model weights saved as model_weights_{args.model}_{date_str}.pth")
    x = torch.randn(1, 10)
    y = model(x)
    dot = make_dot(y, params=dict(model.named_parameters()))

if __name__ == "__main__":
    # Example run:
    # python main.py --model SUBWORDDAN --merge_num 20000
    main()
