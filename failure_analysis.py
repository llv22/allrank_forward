import argparse
import json
import pandas as pd
from pathlib import Path

def statistic_cal(statistics):
    l = len(statistics)
    average = {}
    for v in statistics.values():
        for measure, value in v.items():
            if measure in average:
                average[measure] += value
            else:
                average[measure] = value
    average = {k:v/l for k, v in average.items()}
    return average
    
def p(rank_to_ground_true, rank):
    if rank <= 0:
        print("rank should be larger than 0")
        exit(1)
    cnt = 0
    for i in range(rank):
        if i < len(rank_to_ground_true) and rank_to_ground_true[i] == 1:
            cnt += 1
    return cnt / rank

def mrr(rank_to_ground_true):
    for i in range(len(rank_to_ground_true)):
        if i < len(rank_to_ground_true) and rank_to_ground_true[i] == 1:
            return 1 / (i + 1)
    return 0

import numpy as np

ndcg_k = 5

def dcg(relevances, k=5):
    relevances = np.asfarray(relevances)[:k] if k > 0 else np.asfarray(relevances)
    if relevances.size:
        return np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))
    return 0

def ndcg(relevances, k=5):
    ideal_relevances = sorted(relevances, reverse=True)
    actual_dcg = dcg(relevances, k)
    ideal_dcg = dcg(ideal_relevances, k)
    if ideal_dcg == 0:
        return 0
    return actual_dcg / ideal_dcg

def sort_key_for_dictionary(dic):
    myKeys = list(dic.keys())
    myKeys.sort()
    sorted_dict = {i: dic[i] for i in myKeys}
    return sorted_dict

def convert_to_end_to_end(statistic):
    for k, v, in statistic.items():
        statistic[k] = v
    return statistic

def generate_google_rank_to_ground_true(google_rank, ground_true, max_rank):
    google_rank_to_ground_true = [-1] * (max_rank + 1)
    for i in range(len(google_rank)):
        google_rank_to_ground_true[google_rank[i]] = ground_true[i]
    return google_rank_to_ground_true

def generate_execution_scoring(google_rank, ratings, max_rank):
    google_rank_to_ground_true = [float('-inf')] * (max_rank + 1)
    for i in range(len(google_rank)):
        google_rank_to_ground_true[google_rank[i]] = ratings[i]
    return google_rank_to_ground_true

def generate_execution_scoring_rating_pair(google_rank, ratings, max_rank):
    # google_rank_to_ground_true = [float('-inf')] * (max_rank + 1)
    google_rank_to_ground_true = [(-1, i) for i in range(max_rank + 1)]
    for i in range(len(google_rank)):
        google_rank_to_ground_true[google_rank[i]] = (ratings[i], google_rank[i])
    return google_rank_to_ground_true

def filter_value(statistics):
    filtered_statistics = {}
    for k, v in statistics.items():
        if v['mrr'] != 0 or v['p1'] != 0 or v['p5'] != 0 or v['ndcg1'] != 0 or v[f'ndcg{ndcg_k}'] != 0:
            filtered_statistics[k] = v
    return filtered_statistics

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--exclude_google_rank_for_execution", type=bool, default=False)
    args.add_argument("--rerank_result", type=str, default='experiments/neuralNDCG/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth/results/neuralndcg_atmax_Multimodal_Feature18_label2_on_cohere_ground_truth/predicted_result.txt')
    args.add_argument("--input_label", type=str, default='mmdataset/Feature_18_zeroshot_label2/test.txt')
    args.add_argument("--input_query", type=str, default='mmdataset/Feature_18_zeroshot_label2/test_qid_label2.json')
    args.add_argument("--input_rank", type=str, default='/data/orlando/workspace/metaGUI_forward/repo_upload/ground_truth/execution_result_by_rerank.xlsx')
    conf = args.parse_args()
    group_cnt = 0
    zero_cnt = 0
    non_zero_cnt = 0
    execution_statistics = {}
    google_statistics = {}
    
    qid_to_ranks = {}
    folder = str(Path(conf.rerank_result).parent)
    with open(conf.rerank_result, 'r') as f:
        lines = f.readlines()
        qid = None; ranks = []; line_index = 0;
        for pl, l in enumerate(lines):
            if l.startswith("query_id:"):
                qid = int(l.split(":")[1].strip())
            elif l.strip() == "":
                qid_to_ranks[qid] = ranks
                qid = None
                ranks = []
            else:
                if qid is None:
                    print("error")
                    exit(1)
                else:
                    data = l.strip().split(";")
                    xb = data[0].split(":")[1].strip()[1:-1].split(",")
                    xb =[e.strip() for e in xb]
                    query_str = f"qid:{qid}"
                    for i in range(len(xb)):
                        query_str += f" {i + 1}:{float(xb[i]):.6f}"
                    query_tuple = set([int(qid), float(xb[0]), float(xb[1]), float(xb[2]), float(xb[3]), float(xb[4]), float(xb[5]), float(xb[6]), float(xb[7]), float(xb[8]), float(xb[9]), float(xb[10]), float(xb[11]), float(xb[12]), float(xb[13]), float(xb[14]), float(xb[15]), float(xb[16]), float(xb[17])])
                    scoring = float(data[1].split(":")[1].strip())
                    ground_truth = int(float(data[2].split(":")[1].strip()))
                    ranks.append({
                        "line_index": line_index,
                        "gline": pl,
                        "query_str": query_str,
                        "query_tuple": query_tuple,
                        "scoring": scoring,
                        "ground_truth": ground_truth,
                    })
                    line_index += 1
    query_str_to_rank = {}
    query_str_to_label = {}
    with open(conf.input_label, 'r') as f:
        lines = f.readlines()
        for index, l in enumerate(lines):
            l = l.strip()
            data = l.split(" ")
            label = int(data[0])
            query_str = data[1] + " " + " ".join([f"{index+1}:{float(d.split(':')[1]):.6f}" for index, d in enumerate(data[2:-1])])
            qid = int(data[1].split(":")[1])
            rank = int(data[-1].split(":")[1])
            query_tuple = set([e.split(":")[1] for e in data[1: -1]])
            query_str_to_rank[query_str] = rank
            query_str_to_label[query_str] = label
    with open(conf.input_query, 'r') as f:
        query_to_qid = json.load(f)
        qid_to_query = {v: k for k, v in query_to_qid.items()}
    for qid, ranks in qid_to_ranks.items():
        for r in ranks:
            query_str = r['query_str']
            assert query_str in query_str_to_rank
            assert r['ground_truth'] == query_str_to_label[query_str]
            r['google_rank'] = query_str_to_rank[query_str]
    
    qid_to_failures = {}
    for qid, ranks in qid_to_ranks.items():
        example = []
        for r in ranks:
            if r['ground_truth'] == 0 and r['scoring'] > 0 or r['ground_truth'] == 1 and r['scoring'] < 0:
                example.append({
                    "query_str": r['query_str'],
                    "scoring": r['scoring'],
                    "ground_truth": r['ground_truth'],
                    "google_rank": r['google_rank'],
                })
        if len(example) > 0:
            qid_to_failures[qid] = example
    with open(f"failures.json", 'w') as f:
        json.dump(qid_to_failures, f, indent=4)