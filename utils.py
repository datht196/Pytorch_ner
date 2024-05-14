from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
import importlib
from sklearn.metrics import f1_score



def prepare_conll_data_format(path: str):
    """
    Prepare data in CoNNL like format.
    Tokens and labels separated on each line.
    Sentences are separated by empty line.
    Labels should already be in necessary format, e.g. IO, BIO, BILUO, ...

    Data example:
    token_11    label_11
    token_12    label_12

    token_21    label_21
    token_22    label_22
    token_23    label_23

    ...
    """

    token_seq = []
    label_seq = []
    with open(path, mode="r") as fp:
        tokens = []
        labels = []
        for line in fp:
            if line != "\n":
                pairs = line.strip().split()
                token=pairs[0]
                label=pairs[-1]
                token=token.lower()
                tokens.append(token)
                labels.append(label)
            else:
                if len(tokens) > 0:
                    token_seq.append(tokens)
                    label_seq.append(labels)
                tokens = []
                labels = []

    return token_seq, label_seq


def build_vocab_token(
    train_token: List[List[str]],
    valid_token: List[List[str]],
    test_token: List[List[str]]
    )-> Dict[str, int]:
    
    vocab_token={}
    vocab_token["<PAD>"] = len(vocab_token)
    vocab_token["<UNK>"] = len(vocab_token)
    for sentence in train_token:
        for token in sentence:
            if token not in vocab_token:
                vocab_token[token] = len(vocab_token) + 1
    for sentence in valid_token:
        for token in sentence:
            if token not in vocab_token:
                vocab_token[token] = len(vocab_token) + 1
    for sentence in test_token:
        for token in sentence:
            if token not in vocab_token:
                vocab_token[token] = len(vocab_token) + 1
    return vocab_token
    


def build_vocab_label(
    train_label_seq: List[str]
    ) -> Dict[str, int]:
    """
    Get mapping from labels to indices.
    """


    label_vocab = {'unk':0}
    for sentence in train_label_seq:
        for label in sentence:
            if label not in label_vocab:
                label_vocab[label] = len(label_vocab)
        

    return label_vocab


def process_tokens( 
    tokens: List[str],
    vocab_token: Dict[str,int],
    unk: str = "<UNK>") ->List[int]  :
    """
    Get mapping from labels to indices.
    """
    
    processed_tokens = [vocab_token.get(token, vocab_token[unk]) for token in tokens]
    return processed_tokens

def process_labels(
    labels: List[str],
    vocab_label: Dict[str,int]
    ) -> List[int] :
    """
    Transform list of labels into list of labels' indices.
    """
    processed_labels = [vocab_label[label] for label in labels]
    return processed_labels

def load_word2vec(path: str
    )-> Tuple[Dict[str, int], np.ndarray] :
    """
    Load word2vec embeddings.
    """

    vocab_token={}
    vocab_token["<PAD>"] = len(vocab_token)
    vocab_token["<UNK>"] = len(vocab_token)

    model = KeyedVectors.load_word2vec_format(path,binary=True)
    embedding_dim=model.vector_size
    for token in model.index_to_key:
        vocab_token[token] = len(vocab_token)

    token_embeddings = model.vectors

    unk_embedding = token_embeddings.mean(axis=0)
    token_embeddings = np.vstack([unk_embedding, token_embeddings])
    pad_embedding = np.zeros(shape=token_embeddings.shape[-1])
    token_embeddings = np.vstack([pad_embedding, token_embeddings])

    return embedding_dim, token_embeddings
def str_to_class(module_name, class_name):
    """
    Convert string to Python class object.
    https://stackoverflow.com/questions/1176136/convert-string-to-python-class-object
    """

    # load the module, will raise ImportError if module cannot be loaded
    module = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    cls = getattr(module, class_name)
    return cls
def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Chuyển dữ liệu từ torch.Tensor thành np.ndarray.
    """

    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def calculate_metrics(metrics,loss,y_true,y_pred,idx2label,) :
    

    metrics["loss"].append(loss)

    f1_per_class = f1_score(
        y_true=y_true, y_pred=y_pred, labels=range(len(idx2label)),average=None,zero_division=1
    )
    f1_weighted = f1_score(y_true=y_true, y_pred=y_pred, average="weighted",zero_division=1)
    for cls, f1 in enumerate(f1_per_class):
        metrics[f"f1 {idx2label[cls]}"].append(f1)
    metrics["f1-weighted"].append(f1_weighted)

    return metrics




