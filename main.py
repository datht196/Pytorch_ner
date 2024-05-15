import argparse
import random
import time

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from dataset import NERCollator, NERDataset
from model import Named_Entity_Recognition
from train import train_epoch, validate_epoch
from utils import (
    build_vocab_label, build_vocab_token,
    load_word2vec, prepare_conll_data_format
)

seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Named Entity Recognition Model')
    parser.add_argument('--train_path', default='data/eng.train')
    parser.add_argument('--dev_path', default='data/eng.testa')
    parser.add_argument('--test_path', default='data/eng.testb')
    parser.add_argument('--percentile', type=int, default=100)
    parser.add_argument('--pretrain_embed_path',
                        default="GoogleNews-vectors-negative300.bin")
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.015)
    args = parser.parse_args()

    # Setup device
    DEVICE = "cpu"

    # processing data
    train_token_seq, train_label_seq = prepare_conll_data_format(
        args.train_path)
    valid_token_seq, valid_label_seq = prepare_conll_data_format(args.dev_path)
    test_token_seq, test_label_seq = prepare_conll_data_format(args.test_path)

    # Build vocab_token, vocab_label
    vocab_token = build_vocab_token(
        train_token_seq, valid_token_seq, test_token_seq)
    vocab_label = build_vocab_label(train_label_seq)

    # Data set
    train_set = NERDataset(
        train_token_seq, train_label_seq,
        vocab_token, vocab_label)
    valid_set = NERDataset(
        valid_token_seq, valid_label_seq,
        vocab_token, vocab_label)
    test_set = NERDataset(
        test_token_seq, test_label_seq,
        vocab_token, vocab_label)

    # Collate function
    my_collator = NERCollator(
        vocab_token['<PAD>'], vocab_label['O'], percentile=args.percentile)

    # DataLoader
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, collate_fn=my_collator,)
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size,
        shuffle=False, collate_fn=my_collator)
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size,
        shuffle=False, collate_fn=my_collator)

    # Load word2vec embedding
    embedding_dim, pretrained_embedding = load_word2vec(
        args.pretrain_embed_path)
    vocab_size = len(vocab_token)
    tag_num = len(vocab_label)
    hidden_dim = args.hidden_dim

    # Initialize model, optimizer and loss function
    model = Named_Entity_Recognition(
        vocab_size, embedding_dim, args.hidden_dim,
        pretrained_embedding, tag_num)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction="none")

    start_time = time.time()

    # Training loop
    for epoch in range(args.epochs):
        print(f"epoch [{epoch+1}/{args.epochs}]")
        loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        print("Train loss = ", loss)
        valid_metrics = validate_epoch(model, valid_loader, criterion, DEVICE)

    # Evaluation on test set
    print("Evaluation on test set")
    test_metrics = validate_epoch(model, test_loader, criterion, DEVICE)
    for metric_name, metric_list in test_metrics.items():
        print(f" {metric_name}: {metric_list}")

    # Calculate time
    end_time = time.time()
    train_cost = end_time - start_time
    minute = int(train_cost / 60)
    second = int(train_cost % 60 % 60)
    print('train end', '-' * 50)
    print('train total cost {}m {}s'.format(minute, second))
    print('-' * 50)
