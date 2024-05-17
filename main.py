import argparse  # Library for parsing command-line arguments
import random  # Library for generating random numbers
import time  # Library for handling time-related tasks

import torch  # Library for tensor computations
import numpy as np  # Library for scientific computing
import torch.nn as nn  # Library for neural network modules
import torch.optim as optim  # Library for optimization algorithms
from torch.utils.data import DataLoader  # Library for data loading utilities

# Import custom modules from other files
from dataset import NERCollator, NERDataset
from model import Named_Entity_Recognition
from train import Inference, NERTrainer
from utils import (
    build_vocab_label, build_vocab_token,
    decode_labels, load_word2vec,
    prepare_conll_data_format
)

# Set random seed for reproducibility
seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

if __name__ == '__main__':
    # Setup command-line arguments
    parser = argparse.ArgumentParser(
        description='Named Entity Recognition Model')
    parser.add_argument('--train_path', default='data/eng.train')
    parser.add_argument('--dev_path', default='data/eng.testa')
    parser.add_argument('--test_path', default='data/eng.testb')
    parser.add_argument('--percentile', type=int, default=100)
    parser.add_argument('--pretrain_embed_path',
                        default="GoogleNews-vectors-negative300.bin")
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    # Setup device for computation (CPU or GPU)
    DEVICE = "cpu"

    # Process input data
    train_token_seq, train_label_seq = prepare_conll_data_format(
        args.train_path)
    valid_token_seq, valid_label_seq = prepare_conll_data_format(args.dev_path)
    test_token_seq, test_label_seq = prepare_conll_data_format(args.test_path)

    # Build vocabulary for tokens and labels
    vocab_token = build_vocab_token(
        train_token_seq, valid_token_seq, test_token_seq)
    vocab_label = build_vocab_label(train_label_seq)

    # Create datasets
    train_set = NERDataset(
        train_token_seq, train_label_seq, vocab_token, vocab_label)
    valid_set = NERDataset(
        valid_token_seq, valid_label_seq, vocab_token, vocab_label)
    test_set = NERDataset(test_token_seq, test_label_seq,
                          vocab_token, vocab_label)

    # Collate function for DataLoader
    my_collator = NERCollator(
        vocab_token['<PAD>'], vocab_label['O'], percentile=args.percentile)

    # Create DataLoaders for training, validation, and testing
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=my_collator)
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=my_collator)
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, collate_fn=my_collator)

    # Load pretrained word2vec embeddings
    embedding_dim, pretrained_embedding = load_word2vec(
        args.pretrain_embed_path)
    vocab_size = len(vocab_token)
    tag_num = len(vocab_label)
    hidden_dim = args.hidden_dim

    # Initialize model, optimizer, and loss function
    model = Named_Entity_Recognition(
        vocab_size, embedding_dim, args.hidden_dim, pretrained_embedding, tag_num)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction="none")

    # Create trainer and inference instances
    trainer = NERTrainer(model, train_loader, criterion, optimizer, DEVICE)
    val_inference = Inference(model, valid_loader, DEVICE)
    start_time = time.time()

    # Training loop
    for epoch in range(args.epochs):
        print(f"epoch [{epoch+1}/{args.epochs}]")
        loss = trainer.train_epoch()
        print("Train loss = ", loss)
        val_true_labels, val_pred_labels = val_inference.predict()

    # Evaluation on test set
    print("Evaluation on test set")
    test_inference = Inference(model, test_loader, DEVICE)
    test_true_labels, test_pred_labels = test_inference.predict()

    # Decode label indices to BIOES format
    true_labels_decode = decode_labels(test_true_labels, vocab_label)
    pred_labels_decode = decode_labels(test_pred_labels, vocab_label)

end_time = time.time()
train_cost = end_time - start_time
minute = int(train_cost / 60)
second = int(train_cost % 60 % 60)
print('train end', '-' * 50)
print('train total cost {}m {}s'.format(minute, second))
print('-' * 50)
