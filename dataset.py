import numpy as np
import torch

from typing import Dict, List, Tuple
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset
from utils import process_labels, process_tokens


class NERDataset(Dataset):
    """
    PyTorch Dataset for NER data format.
    Dataset might be preprocessed for more efficiency.
    """

    def __init__(
            self, token_seq: List[List[str]], label_seq: List[List[str]],
            vocab_token: Dict[str, int], vocab_label: Dict[str, int]):
        """
        Mapping token,label to index
        Args:
            token_seq(List[List[str]]): list tokens sequence.
            label_seq(List[List[str]]): list labels sequence.
            vocab_token(Dict[str, int]): dictionary token.
            vocab_label(Dict[str, int]): dictionary label.
        """
        self.vocab_token = vocab_token
        self.vocab_label = vocab_label
        self.token_seq = []
        for tokens in token_seq:
            processed_tokens = process_tokens(tokens, vocab_token)
            self.token_seq.append(processed_tokens)

        self.label_seq = []
        for labels in label_seq:
            processed_labels = process_labels(labels, vocab_label)
            self.label_seq.append(processed_labels)

    def __len__(self):
        return len(self.token_seq)

    def __getitem__(
            self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tokens = self.token_seq[index]
        labels = self.label_seq[index]
        lengths = [len(tokens)]

        return np.array(tokens), np.array(labels), np.array(lengths)


class NERCollator:
    """
    Collator that handles variable-size sentences.
    """

    def __init__(
            self, token_padding_value: int,
            label_padding_value: int, percentile=100):

        self.token_padding_value = token_padding_value
        self.label_padding_value = label_padding_value
        self.percentile = percentile

    def __call__(
        self, batch: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform collation on the batch of data.

        Input:
            batch (List[Tuple[np.ndarray, np.ndarray, np.ndarray]])
            batch = List[Tuple[tokens,labels,lengths]]
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Padded tokens, padded labels, and lengths.
        """

        # Unzip the batch into separate lists of tokens, labels, and lengths
        tokens, labels, lengths = zip(*batch)

        tokens = [list(i) for i in tokens]
        labels = [list(i) for i in labels]

        # Compute the maximum length of sequences based on a percentile
        max_len = int(np.percentile(lengths, self.percentile))

        # Clip lengths to a maximum value
        # Convert them into a PyTorch tensor
        lengths = torch.tensor(
            np.clip(lengths, a_min=0, a_max=max_len),
            dtype=torch.long,
        ).squeeze(-1)

        # Truncate token and label sequences to the maximum length.
        # Convert them into PyTorch tensors.
        for i in range(len(batch)):
            tokens[i] = torch.tensor(tokens[i][:max_len], dtype=torch.long)
            labels[i] = torch.tensor(labels[i][:max_len], dtype=torch.long)
        sorted_index = torch.argsort(lengths, descending=True)

        # Padding tokens
        tokens = pad_sequence(
            tokens, padding_value=self.token_padding_value,
            batch_first=True)[sorted_index]

        # Padding labels
        labels = pad_sequence(
            labels, padding_value=self.label_padding_value,
            batch_first=True)[sorted_index]
        lengths = lengths[sorted_index]

        return tokens, labels, lengths
