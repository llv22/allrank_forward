import numpy as np
import torch

global export_failure
global collect_number
global mrr_f_cnt
global mrr_s_cnt
global ap_f_cnt
global ap_s_cnt
global end_epoch
global query_id

export_failure = False
collect_number = 1000
mrr_f_cnt = 0
mrr_s_cnt = 0
ap_f_cnt = 0
ap_s_cnt = 0
p_f_cnt = 0
p_s_cnt = 0
end_epoch = 19
query_id = 10000000

from allrank.data.dataset_loading import PADDED_Y_VALUE

# see: as we need to make the model to learn order, so we set filler_value = 0.0001 to force the model to learn where to put the positive cases
def ndcg(y_pred, y_true, ats=None, gain_function=lambda x: torch.pow(2, x) - 1, padding_indicator=PADDED_Y_VALUE,
         filler_value=0.0, xb=None, model=None, epoch=None, output_dir=None):
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
    # num_zeros = torch.eq(idcg, 0).sum()
    # print("Number of zeros:", num_zeros.item())

    assert (ndcg_ < 0.0).sum() >= 0, "every ndcg should be non-negative"

    return ndcg_


def __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator=PADDED_Y_VALUE):
    mask = y_true == padding_indicator

    y_pred[mask] = float('-inf')
    y_true[mask] = 0

    desc_y_pred, indices = y_pred.sort(descending=True, dim=-1)
    return torch.gather(y_true, dim=1, index=indices)


def __apply_mask_and_get_indices_and_true_sorted_by_preds(y_pred, y_true, padding_indicator=PADDED_Y_VALUE):
    mask = y_true == padding_indicator

    y_pred[mask] = float('-inf')
    y_true[mask] = 0

    _, indices = y_pred.sort(descending=True, dim=-1)
    return indices, torch.gather(y_true, dim=1, index=indices)


def dcg(y_pred, y_true, ats=None, gain_function=lambda x: torch.pow(2, x) - 1, padding_indicator=PADDED_Y_VALUE, output_dir=None):
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

    
def output_to_file(file_name, y_pred, y_true, xb, epoch, start_index):
    def format_array(arr):
        return [format(x, '.4f') for x in arr]
    with open(file_name, 'a') as f:
        f.write("epoch: " + str(epoch) + "\n")
        for i in range(y_pred.shape[0]):
            f.write("case: " + str(start_index + i) + "\n")
            f.write("xb: " + str(format_array(xb[i].tolist())) + "\n")
            f.write("y_pred: " + str(y_pred[i].tolist()) + "\n")
            f.write("y_true: " + str(y_true[i].tolist()) + "\n")
            f.write("\n")

def output_to_file_p(file_name, y_pred, y_true, xb):
    def format_array(arr):
        return [float(format(x, '.6f')) for x in arr]
    global query_id
    with open(file_name, 'a') as f:
        for i in range(y_pred.shape[0]):
            f.write(f"query_id: {query_id}\n")
            for j in range(y_pred.shape[1]):
                if y_pred[i][j] != float('-inf') and y_true[i][j] >= 0:
                    f.write(f"xb: {format_array(xb[i][j].tolist())}; y_pred: {format(y_pred[i][j].item(), '.6f')}; y_true: {format(y_true[i][j].item(), '.6f')}\n")
            f.write("\n")
            query_id += 1

def mrr(y_pred, y_true, ats=None, padding_indicator=PADDED_Y_VALUE, xb=None, model=None, epoch=None, output_dir=None):
    """
    Mean Reciprocal Rank at k.

    Compute MRR at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for MRR evaluation, if None, maximum rank is used
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: MRR values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    if export_failure and not model.training:
        # see: neuralNDCG
        # mrr_index = 0
        # meta_setting = 0.18376069
        # threshold_success = meta_setting * 0.8 #
        # threshold_failure = meta_setting * 0.2 # 
        # success_cases = (result > threshold_success)[:,0]
        # global mrr_f_cnt, mrr_s_cnt
        # if mrr_s_cnt < collect_number and y_pred[success_cases, mrr_index].shape[0] > 0:
        #     if y_pred[success_cases, mrr_index].shape[0] - mrr_s_cnt >= collect_number:
        #         # output the first [0, collect_number] success cases
        #         take_number = collect_number - mrr_s_cnt
        #         output_to_file("mrr_success_cases.txt", y_pred[success_cases, mrr_index][:take_number], y_true[success_cases, mrr_index][:take_number], xb[success_cases, mrr_index, :][:take_number,:], epoch, mrr_s_cnt)
        #         mrr_s_cnt = collect_number
        #     else:
        #         # output the first [0, collect_number] success cases
        #         output_to_file("mrr_success_cases.txt", y_pred[success_cases, mrr_index], y_true[success_cases, mrr_index], xb[success_cases, mrr_index, :], epoch, mrr_s_cnt)
        #         mrr_s_cnt += success_cases.shape[0]
        # failure_cases = (result <= threshold_failure)[:,0]
        # if mrr_f_cnt < collect_number and y_pred[failure_cases, mrr_index].shape[0] > 0:
        #     if y_pred[failure_cases, mrr_index].shape[0] - mrr_f_cnt >= collect_number:
        #         take_number = collect_number - mrr_f_cnt
        #         output_to_file("mrr_failure_cases.txt", y_pred[failure_cases, mrr_index][:take_number], y_true[failure_cases, mrr_index][:take_number], xb[failure_cases, mrr_index, :][:take_number,:], epoch, mrr_f_cnt)
        #         mrr_f_cnt = collect_number
        #     else:
        #         # output the first [0, collect_number] failure cases
        #         output_to_file("mrr_failure_cases.txt", y_pred[failure_cases, mrr_index], y_true[failure_cases, mrr_index], xb[failure_cases, mrr_index, :], epoch, mrr_f_cnt)
        #         mrr_f_cnt += y_pred[failure_cases, mrr_index].shape[0]
        # see: export for checking results
        if output_dir is not None:
            f = f"{output_dir}/predicted_result.txt"
        else:
            f = "predicted_result.txt"
        output_to_file_p(f, y_pred, y_true, xb)
        
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    if ats is None:
        ats = [y_true.shape[1]]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator)

    values, indices = torch.max(true_sorted_by_preds, dim=1)
    # see: extend indices to shape [batch_size, len(ats)]
    indices = indices.type_as(values).unsqueeze(dim=0).t().expand(len(y_true), len(ats))
    ats_rep = torch.tensor(data=ats, device=indices.device, dtype=torch.float32).expand(len(y_true), len(ats))

    within_at_mask = (indices < ats_rep).type(torch.float32)

    result = torch.tensor(1.0) / (indices + torch.tensor(1.0))

    # see: because we use 1/(1+rank) as the reciprocal rank, result is always != 0. However, if all true labels are 0, we need to hardcode the reciprocal rank to 0
    ## see: orlando fuxxk your implementation by incorrect masking
    # zero_sum_mask = torch.sum(values) == 0.0
    zero_sum_mask = values == 0.0
    zero_sum_mask = zero_sum_mask.unsqueeze(dim=1).expand(len(y_true), len(ats))
    result[zero_sum_mask] = 0.0

    result = result * within_at_mask
    
    return result


def precision(y_pred, y_true, ats=None, padding_indicator=PADDED_Y_VALUE, xb=None, model=None, epoch=None, output_dir=None):
    y_true = y_true.clone()
    y_pred = y_pred.clone()
    
    if ats is None:
        ats = [y_true.shape[1]]

    # Apply mask and get true labels sorted by predicted scores
    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator)

    # Compute precision at each cutoff from 1 to k and average
    precisions = torch.zeros(len(y_true), len(ats), dtype=torch.float32, device=y_true.device)
    for index, i in enumerate(ats):
        top_k = true_sorted_by_preds[:, :i]  # Slice top k predictions
        top_k_hits = (top_k > 0)  # Create indicator matrix for top k predictions
        average = torch.sum(top_k_hits, dim=-1, dtype=torch.float32)  # Compute hits
        # top_k_cumsum_hits = torch.cumsum(top_k_hits, dim=1)
        precisions[:, index] = average / i
    
    assert not torch.isnan(precisions).any(), "precision should not be nan"
    assert (precisions < 0.0).sum() >= 0, "every precisions should be non-negative"
    assert (precisions > 1.0).sum() >= 0, "every precisions should be less-equal than 1"
    
    # if export_failure and not model.training:  
    #     # see: neuralNDCG
    #     result = precisions
    #     meta_settings = [0.17948718, 0.104273506]
    #     threshold_successes = [i * 0.7 for i in meta_settings] #
    #     threshold_failures = [i * 0.2 for i in meta_settings]#       
    #     result = precisions
    #     global ap_f_cnt, ap_s_cnt
    #     p_indexs = [0, 1]
    #     for p_index in p_indexs:
    #         if p_index == 0:
    #             ap_success_cases_f = "p1_success_cases.txt"
    #             ap_failure_cases_f = "p1_failure_cases.txt"
    #         else:    
    #             ap_success_cases_f = "p5_success_cases.txt"
    #             ap_failure_cases_f = "p5_failure_cases.txt"
    #         threshold_success = threshold_successes[p_index]
    #         success_cases = (result > threshold_success)[:,0]
    #         if ap_s_cnt < collect_number and y_pred[success_cases, p_index].shape[0] > 0:
    #             if y_pred[success_cases, p_index].shape[0] - ap_s_cnt >= collect_number:
    #                 # output the first [0, collect_number] success cases
    #                 take_number = collect_number - ap_s_cnt
    #                 output_to_file(ap_success_cases_f, y_pred[success_cases, p_index][:take_number], y_true[success_cases, p_index][:take_number], xb[success_cases, p_index, :][:take_number,:], epoch, ap_s_cnt)
    #                 ap_s_cnt = collect_number
    #             else:
    #                 # output the first [0, collect_number] success cases
    #                 output_to_file(ap_success_cases_f, y_pred[success_cases, p_index], y_true[success_cases, p_index], xb[success_cases, p_index, :], epoch, ap_s_cnt)
    #                 ap_s_cnt += success_cases.shape[0]
    #         threshold_failure = threshold_failures[p_index]
    #         failure_cases = (result <= threshold_failure)[:,0]
    #         if ap_f_cnt < collect_number and y_pred[failure_cases, p_index].shape[0] > 0:
    #             if y_pred[failure_cases, p_index].shape[0] - ap_f_cnt >= collect_number:
    #                 take_number = collect_number - ap_f_cnt
    #                 output_to_file(ap_failure_cases_f, y_pred[failure_cases, p_index][:take_number], y_true[failure_cases, p_index][:take_number], xb[failure_cases, p_index, :][:take_number,:], epoch, ap_f_cnt)
    #                 ap_f_cnt = collect_number
    #             else:
    #                 # output the first [0, collect_number] failure cases
    #                 output_to_file(ap_failure_cases_f, y_pred[failure_cases, p_index], y_true[failure_cases, p_index], xb[failure_cases, p_index, :], epoch, ap_f_cnt)
    #                 ap_f_cnt += y_pred[failure_cases, p_index].shape[0]


    return precisions

def ap(y_pred, y_true, ats=None, padding_indicator=PADDED_Y_VALUE, xb=None, model=None, epoch=None, output_dir=None):
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