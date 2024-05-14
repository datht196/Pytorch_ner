import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable
from typing import Callable, DefaultDict, List, Optional
from collections import defaultdict
from utils import to_numpy,calculate_metrics
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
def masking(lengths: torch.Tensor) -> torch.Tensor:
    
    device = lengths.device
    lengths_shape = lengths.shape[0]
    max_len = lengths.max()
    return torch.arange(end=max_len, device=device).expand(
        size=(lengths_shape, max_len)
    ) < lengths.unsqueeze(1)

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    optimizer: optim.Optimizer,
    device: torch.device,
    )-> DefaultDict[str, List[float]]:
    model.train()
    
    for tokens, labels, lengths in dataloader:
        tokens, labels, lengths = (
            tokens.to(device),
            labels.to(device),
            lengths.to(device),
        )
        model.zero_grad()
        mask = masking(lengths)

        #Forward pass
        logits = model(tokens, lengths)
        loss_without_reduction = criterion(logits.transpose(-1, -2), labels)
        loss = torch.sum(loss_without_reduction * mask) / torch.sum(mask)
        
        # Backward
        
        loss.backward()
        
        
        #gradient_clipping
        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=0.1,
            norm_type=2,
        )
        
        optimizer.step()
        

    return loss

def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    device: torch.device,
    ) -> DefaultDict[str, List[float]]:

    metrics= defaultdict(list)
    idx2label = {v: k for k, v in dataloader.dataset.vocab_label.items()}

    model.eval()
    y_true_accumulator = []
    y_pred_accumulator = []
    
    
    for tokens, labels, lengths in dataloader:
        tokens, labels, lengths = (
            tokens.to(device),
            labels.to(device),
            lengths.to(device),
        )

        mask = masking(lengths)

        
        # Forward pass
        with torch.no_grad():
            logits = model(tokens, lengths)
            loss_without_reduction = criterion(logits.transpose(-1, -2), labels)
            loss = torch.sum(loss_without_reduction * mask) / torch.sum(mask)
        
        # Make prediction
        y_true=to_numpy(labels[mask])
        y_pred=to_numpy(logits.argmax(dim=-1)[mask])
        y_true_accumulator += y_true.tolist()
        y_pred_accumulator += y_pred.tolist()
        
    # Metrics
    metrics = calculate_metrics(
        metrics=metrics,
        loss=loss.item(),
        y_true=y_true_accumulator,
        y_pred=y_pred_accumulator,
        idx2label=idx2label,
    ) 
    return metrics

