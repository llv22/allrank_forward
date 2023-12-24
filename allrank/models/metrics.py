import numpy as np
import torch

from allrank.data.dataset_loading import PADDED_Y_VALUE


def ndcg(y_pred, y_true, ats=None, gain_function=lambda x: torch.pow(2, x) - 1, padding_indicator=PADDED_Y_VALUE,
         filler_value=1.0):
    """
    Normalized Discounted Cumulative Gain at k.

    Compute NDCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for NDCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param filler_value: a filler NDCG value to use when there are no relevant items in listing
    :return: NDCG values for each slate and rank passed, shape [batch_size, len(ats)]
    """
    idcg = dcg(y_true, y_true, ats, gain_function, padding_indicator)
    ndcg_ = dcg(y_pred, y_true, ats, gain_function, padding_indicator) / idcg
    idcg_mask = idcg == 0
    ndcg_[idcg_mask] = filler_value  # if idcg == 0 , set ndcg to filler_value

    assert (ndcg_ < 0.0).sum() >= 0, "every ndcg should be non-negative"

    return ndcg_


def __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator=PADDED_Y_VALUE):
    mask = y_true == padding_indicator

    y_pred[mask] = float('-inf')
    y_true[mask] = 0.0

    _, indices = y_pred.sort(descending=True, dim=-1)
    return torch.gather(y_true, dim=1, index=indices)


def __apply_mask_and_get_indices_and_true_sorted_by_preds(y_pred, y_true, padding_indicator=PADDED_Y_VALUE):
    mask = y_true == padding_indicator

    y_pred[mask] = float('-inf')
    y_true[mask] = 0.0

    _, indices = y_pred.sort(descending=True, dim=-1)
    return indices, torch.gather(y_true, dim=1, index=indices)


def dcg(y_pred, y_true, ats=None, gain_function=lambda x: torch.pow(2, x) - 1, padding_indicator=PADDED_Y_VALUE):
    """
    Discounted Cumulative Gain at k.

    Compute DCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for DCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: DCG values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    actual_length = y_true.shape[1]

    if ats is None:
        ats = [actual_length]
    ats = [min(at, actual_length) for at in ats]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator)

    discounts = (torch.tensor(1) / torch.log2(torch.arange(true_sorted_by_preds.shape[1], dtype=torch.float) + 2.0)).to(
        device=true_sorted_by_preds.device)

    gains = gain_function(true_sorted_by_preds)

    discounted_gains = (gains * discounts)[:, :np.max(ats)]

    cum_dcg = torch.cumsum(discounted_gains, dim=1)

    ats_tensor = torch.tensor(ats, dtype=torch.long) - torch.tensor(1)

    dcg = cum_dcg[:, ats_tensor]

    return dcg


def mrr(y_pred, y_true, ats=None, padding_indicator=PADDED_Y_VALUE):
    """
    Mean Reciprocal Rank at k.

    Compute MRR at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for MRR evaluation, if None, maximum rank is used
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: MRR values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    if ats is None:
        ats = [y_true.shape[1]]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator)

    values, indices = torch.max(true_sorted_by_preds, dim=1)
    indices = indices.type_as(values).unsqueeze(dim=0).t().expand(len(y_true), len(ats))

    ats_rep = torch.tensor(data=ats, device=indices.device, dtype=torch.float32).expand(len(y_true), len(ats))

    within_at_mask = (indices < ats_rep).type(torch.float32)

    result = torch.tensor(1.0) / (indices + torch.tensor(1.0))

    # see: because we use 1/(1+rank) as the reciprocal rank, result is always != 0. However, if all true labels are 0, we need to hardcode the reciprocal rank to 0
    zero_sum_mask = torch.sum(values) == 0.0
    result[zero_sum_mask] = 0.0

    result = result * within_at_mask

    return result

def ap(y_pred, y_true, ats=None, padding_indicator=PADDED_Y_VALUE):
    y_true = y_true.clone()
    y_pred = y_pred.clone()
    
    if ats is None:
        ats = [y_true.shape[1]]

    # Apply mask and get true labels sorted by predicted scores
    indices, true_sorted_by_preds = __apply_mask_and_get_indices_and_true_sorted_by_preds(y_pred, y_true, padding_indicator)

    # Compute precision at each cutoff from 1 to k and average
    avg_precisions = torch.zeros(len(y_true), len(ats), dtype=torch.float32, device=y_true.device)
    for index, i in enumerate(ats):
        top_k = true_sorted_by_preds[:, :i]  # Slice top k predictions
        top_k_hits = (top_k > 0)  # Create indicator matrix for top k predictions
        average = torch.sum(top_k_hits, dim=1, dtype=torch.float32)  # Compute hits
        top_k_cumsum_hits = torch.cumsum(top_k_hits, dim=1)
        top_k_cumsum_factors = torch.arange(1, i + 1, dtype=torch.float32, device=top_k_cumsum_hits.device).expand(len(y_true), i)
        top_k_cumsum_hits = top_k_cumsum_hits / top_k_cumsum_factors  # Compute precision at each rank
        precision_sum = torch.sum(top_k_cumsum_hits * top_k_hits, dim=1, dtype=torch.float32)
        non_zero_mask = average != 0
        precision_at_i = torch.zeros_like(precision_sum)
        precision_at_i[non_zero_mask] = precision_sum[non_zero_mask] / average[non_zero_mask]  # Precision at this cutoff
        if torch.isnan(precision_at_i).any():
            print("precision_at_i is nan")
        avg_precisions[:, index] = precision_at_i
    
    assert not torch.isnan(avg_precisions).any(), "avg_precisions should not be nan"
    assert (avg_precisions < 0.0).sum() >= 0, "every avg_precisions should be non-negative"
    assert (avg_precisions > 1.0).sum() >= 0, "every avg_precisions should be less-equal than 1"

    return avg_precisions
    

# def precision(y_pred, y_true, ats=None, padding_indicator=PADDED_Y_VALUE):
#     return 1
    
# def recall(y_pred, y_true, ats=None, padding_indicator=PADDED_Y_VALUE):
#     return 1
    
# def f1(y_pred, y_true, ats=None, padding_indicator=PADDED_Y_VALUE):
#     return 1