from collections import Counter
from typing import Any, Dict
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from train import train_epoch,masking,validate_epoch
from dataset import NERCollator,NERDataset
import numpy as np
import time

from model import LinearHead,DynamicRNN,EmbeddingPreTrained,LSTM
from utils import (
    build_vocab_token,
    build_vocab_label,
    prepare_conll_data_format,
    load_word2vec,
    str_to_class
)
# Setup device
device =  "cpu"

# processing data
train_token_seq, train_label_seq = prepare_conll_data_format(
        path="data/eng.train"
    )
valid_token_seq, valid_label_seq = prepare_conll_data_format(
        path="data/eng.testa"
    )
test_token_seq, test_label_seq = prepare_conll_data_format(
        path="data/eng.testb"
    )

# Build vocab_label,vocab_label
vocab_token=build_vocab_token(train_token_seq,valid_token_seq,test_token_seq)
vocab_label=build_vocab_label(train_label_seq)

#Data set
train_set = NERDataset(train_token_seq,train_label_seq,vocab_token,vocab_label)
valid_set = NERDataset(valid_token_seq,valid_label_seq,vocab_token,vocab_label)
test_set = NERDataset(test_token_seq,test_label_seq,vocab_token,vocab_label)

#Collate_fn
train_collator = NERCollator(vocab_token['<PAD>'],vocab_label['O'],percentile=100)
valid_collator = NERCollator(vocab_token['<PAD>'],vocab_label['O'],percentile=100)
test_collator = NERCollator(vocab_token['<PAD>'],vocab_label['O'],percentile=100)

#data loader
train_loader = DataLoader(train_set,batch_size= 10,shuffle =True,collate_fn= train_collator,)
valid_loader = DataLoader(valid_set,batch_size= 1,shuffle =False,collate_fn= valid_collator)
test_loader = DataLoader(test_set,batch_size= 1,shuffle= False,collate_fn= test_collator)

embedding_dim,pretrained_embedding=load_word2vec("GoogleNews-vectors-negative300.bin")


vocab_size=len(vocab_token)
tagset_size=len(vocab_label)
hidden_dim=64
embedding_layer =EmbeddingPreTrained(pretrained_embedding)

rnn_layer = DynamicRNN(
        rnn_unit=nn.LSTM,
        input_size=embedding_dim,  
        hidden_size=64,
        num_layers=1,
        dropout=0,
        bidirectional=False,
    )

linear_head = LinearHead(
        linear_head=nn.Linear(
            in_features=64,
            out_features=tagset_size,
        )
    )


model = LSTM(
        embedding_layer=embedding_layer,
        rnn_layer=rnn_layer,
        linear_head=linear_head,
    ).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
start_time = time.time()

number_epoch=10#Number epoch
criterion = nn.CrossEntropyLoss(reduction="none")#Loss function

for epoch in range(number_epoch):
    print(f"epoch [{epoch+1}/{number_epoch}]\n")
    loss=train_epoch(model,train_loader,criterion,optimizer,device)
    print("Train loss =",loss)
    valid_metrics=validate_epoch(model,valid_loader,criterion,device)
print("Đánh giá trên tập test")

test_metrics=validate_epoch(model,test_loader,criterion,device)
for metric_name, metric_list in test_metrics.items():
    print(f" {metric_name}: {metric_list}")   

end_time = time.time()
execution_time = end_time - start_time
print("Thời gian chạy:", execution_time, "seconds")

    