import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from typing import Callable, DefaultDict, List, Optional
from torch.utils.data import DataLoader


class EmbeddingPreTrained(nn.Module):
    """
    Init layer embedding word2vec
    """

    def __init__(self, embedding_matrix: np.ndarray):
        super(EmbeddingPreTrained, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(
            torch.Tensor(embedding_matrix).float(),
            freeze=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)
class DynamicRNN(nn.Module):
    """
    Init RNN Layer 
    """

    def __init__(
        self,
        rnn_unit: nn.Module,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool):
        super(DynamicRNN, self).__init__()
        self.rnn = rnn_unit(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_length: torch.Tensor
        )-> torch.Tensor:
        packed_x = pack_padded_sequence(
            x, x_length.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_rnn_out, _ = self.rnn(packed_x)
        rnn_out, _ = pad_packed_sequence(packed_rnn_out, batch_first=True)
        return rnn_out
class LinearHead(nn.Module):
    """
    Init Linear layer .
    """

    def __init__(self, linear_head):
        super(LinearHead, self).__init__()
        self.linear_head = linear_head

    def forward(self, x) :
        return self.linear_head(x)

class LSTM(nn.Module):
    

    def __init__(
        self,
        embedding_layer: nn.Module,
        rnn_layer: nn.Module,
        linear_head: nn.Module,
    ):
        super(LSTM, self).__init__()
        self.embedding = embedding_layer  # EMBEDDINGS
        self.rnn = rnn_layer  # RNN
        self.linear_head = linear_head  # LINEAR HEAD

    def forward(
        self,
        x: torch.Tensor,
        x_length: torch.Tensor
        ) -> torch.Tensor:
        embed = self.embedding(x)  # EMBEDDINGS
        rnn_out = self.rnn(embed, x_length)  # RNN
        logits = self.linear_head(rnn_out)  # LINEAR HEAD
        return logits
