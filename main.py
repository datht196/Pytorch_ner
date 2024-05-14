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

from model import LinearHead,DynamicRNN,EmbeddingPreTrained,BiLSTM
from utils import (
    build_vocab_token,
    build_vocab_label,
    prepare_conll_data_format,
    load_word2vec,
    lr_decay,
    str_to_class
)
# Cài đặt thiết bị
device =  "cpu"

# Tiền xử lý dữ liệu
train_token_seq, train_label_seq = prepare_conll_data_format(
        path="C:/Users/hoang/OneDrive - Hanoi University of Science and Technology/pytorch-NER-master/data/eng.train"
    )
valid_token_seq, valid_label_seq = prepare_conll_data_format(
        path="C:/Users/hoang/OneDrive - Hanoi University of Science and Technology/pytorch-NER-master/data/eng.testa"
    )
test_token_seq, test_label_seq = prepare_conll_data_format(
        path="C:/Users/hoang/OneDrive - Hanoi University of Science and Technology/pytorch-NER-master/data/eng.testb"
    )

# Tạo từ điển token,từ điển label
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

# Chiều của vecto embedding,ma trận embedding
embedding_dim,pretrained_embedding=load_word2vec("G:/My Drive/Data/GoogleNews-vectors-negative300.bin")

vocab_size=len(vocab_token)
tagset_size=len(vocab_label)
hidden_dim=64
print("ok")
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


model = BiLSTM(
        embedding_layer=embedding_layer,
        rnn_layer=rnn_layer,
        linear_head=linear_head,
    ).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)#Thuật toán tối ưu
start_time = time.time()
# Số lượng epoch
number_epoch=10

criterion = nn.CrossEntropyLoss(reduction="none")#Hàm mất mát

for epoch in range(number_epoch):
    print(f"epoch [{epoch+1}/{number_epoch}]\n")
    loss=train_epoch(train_loader,model,optimizer,criterion,device)
    print("Train loss =",loss)
    valid_metrics=validate_epoch(valid_loader,model,criterion,device)
print("Đánh giá trên tập test")

test_metrics=validate_epoch(test_loader,model,criterion,device)
for metric_name, metric_list in test_metrics.items():
    print(f" {metric_name}: {metric_list}")   

end_time = time.time()
execution_time = end_time - start_time
print("Thời gian chạy:", execution_time, "seconds")

    