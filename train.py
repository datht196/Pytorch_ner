from collections import defaultdict
from typing import Callable, DefaultDict, List

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from utils import calculate_metrics, to_numpy


def masking(lengths: torch.Tensor) -> torch.Tensor:
    """
    Creates a boolean mask tensor from a tensor of sequence lengths.
    Input:
        lengths (torch.Tensor): A tensor containing the lengths of sequences
        in a batch. Shape: (batch_size,)
    Output:
        torch.Tensor: A boolean tensor where each element is True
        if the corresponding element in the sequence is within the valid length
        and False otherwise. Shape: (batch_size, max_len)
    """
    device = lengths.device
    lengths_shape = lengths.shape[0]
    max_len = lengths.max()
    mask = torch.arange(end=max_len, device=device).expand(
        size=(lengths_shape, max_len)
    ) < lengths.unsqueeze(1)
    return mask


def train_epoch(
    model: nn.Module, dataloader: DataLoader,
    criterion: Callable, optimizer: optim.Optimizer,
    DEVICE: torch.device,
) -> DefaultDict[str, List[float]]:
    """
    Train the model for one epoch.
    Input:
        model (nn.Module): The PyTorch model to train.
        dataloader (DataLoader): DataLoader providing training data in batches.
        criterion (Callable): The loss function to compute the model's loss.
        optimizer (optim.Optimizer): The optimizer to update the
        model's parameters.
        DEVICE (torch.device): The device (CPU or GPU) to perform computations.
    Output:
        DefaultDict[str, List[float]]: Training results containing metrics
        such as loss and other metrics.
    """
    model.train()
    for tokens, labels, lengths in dataloader:
        tokens, labels, lengths = (
            tokens.to(DEVICE),
            labels.to(DEVICE),
            lengths.to(DEVICE),
        )
        # Zero out the gradients of the model
        model.zero_grad()

        # Create a mask based on the lengths of sequences
        mask = masking(lengths)

        # Forward pass: Compute the model predictions
        logits = model(tokens, lengths)

        # Compute the loss
        loss_without_reduction = criterion(logits.transpose(-1, -2), labels)
        loss = torch.sum(loss_without_reduction * mask) / torch.sum(mask)

        # Backward pass: Compute the gradients
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=0.1,
            norm_type=2,
        )
        # Update the model parameters
        optimizer.step()

    return loss


def validate_epoch(
    model: nn.Module, dataloader: DataLoader,
    criterion: Callable, DEVICE: torch.device,
) -> DefaultDict[str, List[float]]:
    """
    Validate the model on one epoch of validation data.

    Input:
        model (nn.Module): The PyTorch model to validate.
        dataloader (DataLoader): DataLoader providing validation data in batch.
        criterion (Callable): The loss function to compute the model's loss.
        device (torch.device): The device (CPU or GPU) to perform computations.
    Output:
        DefaultDict[str, List[float]]: Validation results containing metrics
        such as loss and other metrics.
    """
    # Initialize metrics and index-to-label mapping
    metrics = defaultdict(list)
    idx2label = {v: k for k, v in dataloader.dataset.vocab_label.items()}

    # Set the model to evaluation mode
    model.eval()

    # Accumulators for true and predicted labels
    y_true_accumulator = []
    y_pred_accumulator = []

    for tokens, labels, lengths in dataloader:
        tokens, labels, lengths = (
            tokens.to(DEVICE),
            labels.to(DEVICE),
            lengths.to(DEVICE),
        )

        # Create a mask based on the lengths of sequences
        mask = masking(lengths)

        # Forward pass
        with torch.no_grad():
            logits = model(tokens, lengths)
            loss_without_reduction = criterion(
                logits.transpose(-1, -2), labels)
            loss = torch.sum(loss_without_reduction * mask) / torch.sum(mask)

        # Make predictions
        y_true = to_numpy(labels[mask])
        y_pred = to_numpy(logits.argmax(dim=-1)[mask])
        y_true_accumulator += y_true.tolist()
        y_pred_accumulator += y_pred.tolist()

    # Calculate metrics
    metrics = calculate_metrics(
        metrics=metrics,
        loss=loss.item(),
        y_true=y_true_accumulator,
        y_pred=y_pred_accumulator,
        idx2label=idx2label,
    )
    return metrics
