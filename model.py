import numpy as np
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Named_Entity_Recognition(nn.Module):
    def __init__(
            self, vocab_size: int, embedding_dim: int, hidden_dim: int,
            pretrained_embedding: np.ndarray, tag_num: int):
        """
        Initialize the NER model.

        Args:
            vocab_size (int): The size of the dictionary.
            embedding_dim (int): The dimension of word embeddings.
            hidden_dim (int): The dimension of hidden states in the LSTM.
            pretrained_embedding (np.ndarray): matrix embeddings.
            tag_num (int): The number of tags .
        """
        super(Named_Entity_Recognition, self).__init__()
        self.input_dim = embedding_dim
        # Embedding layer with pretrained embeddings
        self.embedding = nn.Embedding.from_pretrained(
            torch.Tensor(pretrained_embedding).float(), freeze=True)

        # LSTM layer
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, batch_first=True)

        # Linear layer
        self.hidden2tag = nn.Linear(hidden_dim, tag_num)

    def forward(self, tokens: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NER model.

        Input:
            tokens (torch.Tensor): Input tokens (batch_size x max_seq_length).
            lengths (torch.Tensor): Lengths of input sequences.

        Returns:
            torch.Tensor: Output logits (batch_size x max_seq_length x tag_num)
        """
        # Embedding
        print(tokens.size)
        word_embedding = self.embedding(tokens)
        # LSTM
        packed_words = pack_padded_sequence(
            word_embedding, lengths, batch_first=True, enforce_sorted=True)
        lstm_out, _ = self.lstm(packed_words)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        # Linear
        logits = self.hidden2tag(lstm_out)
        print(logits.size())
        return logits
