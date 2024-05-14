from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from utils import process_labels, process_tokens


class NERDataset(Dataset):
    """
    PyTorch Dataset for NER data format.
    Dataset might be preprocessed for more efficiency.
    """

    def __init__(self,token_seq,label_seq,vocab_token,vocab_label):
        self.vocab_token = vocab_token
        self.vocab_label = vocab_label
        

        self.token_seq = [process_tokens(tokens, vocab_token) for tokens in token_seq]
        self.label_seq = [process_labels(labels, vocab_label) for labels in label_seq]


    def __len__(self):
        return len(self.token_seq)

    def __getitem__(self, index) :
        tokens = self.token_seq[index]
        labels = self.label_seq[index]
        lengths = [len(tokens)]

        return np.array(tokens), np.array(labels), np.array(lengths)

class NERCollator:
    """
    Collator that handles variable-size sentences.
    """
    def __init__(
        self,
        token_padding_value: int,
        label_padding_value: int ,
        percentile= 100,
    ):
        self.token_padding_value = token_padding_value
        self.label_padding_value = label_padding_value
        self.percentile = percentile

    def __call__(
        self,
        batch: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        tokens, labels, lengths = zip(*batch)

        tokens = [list(i) for i in tokens]
        labels = [list(i) for i in labels]

        max_len = int(np.percentile(lengths, self.percentile))

        lengths = torch.tensor(
            np.clip(lengths, a_min=0, a_max=max_len),
            dtype=torch.long,
        ).squeeze(-1)

        for i in range(len(batch)):
            tokens[i] = torch.tensor(tokens[i][:max_len], dtype=torch.long)
            labels[i] = torch.tensor(labels[i][:max_len], dtype=torch.long)

        sorted_index = torch.argsort(lengths, descending=True)

        tokens = pad_sequence(
            tokens, padding_value=self.token_padding_value, batch_first=True
        )[sorted_index]
        labels = pad_sequence(
            labels, padding_value=self.label_padding_value, batch_first=True
        )[sorted_index]
        lengths = lengths[sorted_index]

        return tokens, labels, lengths
        
