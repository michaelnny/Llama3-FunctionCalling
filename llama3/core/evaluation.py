# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

from typing import Tuple
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

# from rouge_score import rouge_scorer


def masked_cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1, reduction='none')
    losses *= loss_mask.float().view(-1)
    loss = losses.sum() / loss_mask.sum()
    return loss


def compute_perplexity(loss: torch.Tensor) -> float:
    return torch.exp(loss.detach()).item()


def compute_masked_accuracy(logits: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = (preds == labels) * loss_mask
    accuracy = correct.sum().float() / loss_mask.sum()
    return accuracy.item()


def compute_masked_f1_score(logits: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor) -> Tuple[float, float, float]:

    predicted = torch.argmax(logits, dim=-1)
    predicted = predicted.view(-1).cpu().numpy()
    labels = labels.view(-1).cpu().numpy()
    mask = loss_mask.view(-1).cpu().numpy()

    # Filter out the ignored indices
    filtered_predictions = predicted[mask == 1]
    filtered_labels = labels[mask == 1]

    precision, recall, f1, _ = precision_recall_fscore_support(filtered_labels, filtered_predictions, average='weighted', zero_division=1)
    return (precision, recall, f1)


def compute_evaluation_metrics(loss: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor, prefix: str = '') -> dict:
    loss = loss.detach()
    logits = logits.detach()
    labels = labels.detach()
    loss_mask = loss_mask.detach()

    perplexity = compute_perplexity(loss)
    accuracy = compute_masked_accuracy(logits, labels, loss_mask)
    precision, recall, f1 = compute_masked_f1_score(logits, labels, loss_mask)

    return {
        f'{prefix}loss': loss.item(),
        f'{prefix}perplexity': perplexity,
        f'{prefix}accuracy': accuracy,
        f'{prefix}precision': precision,
        f'{prefix}recall': recall,
        f'{prefix}f1_score': f1,
    }


# def compute_masked_rouge_score(predictions: torch.Tensor, references: torch.Tensor, loss_mask: torch.Tensor) -> Tuple[float, float]:
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
#     rouge1_scores = []
#     rougeL_scores = []

#     for pred, ref, mask in zip(predictions, references, loss_mask):
#         pred_tokens = pred[mask == 1].tolist()
#         ref_tokens = ref[mask == 1].tolist()

#         pred_text = " ".join(map(str, pred_tokens))
#         ref_text = " ".join(map(str, ref_tokens))

#         scores = scorer.score(ref_text, pred_text)
#         rouge1_scores.append(scores['rouge1'].fmeasure)
#         rougeL_scores.append(scores['rougeL'].fmeasure)

#     avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
#     avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

#     return avg_rouge1, avg_rougeL
